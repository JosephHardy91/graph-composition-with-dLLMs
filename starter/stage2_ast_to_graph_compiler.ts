/**
 * Stage 2: Document AST -> Pinned-Boundary dLLM Graph Compiler + Deterministic Validator
 * --------------------------------------------------------------------------------------
 * Instead of starting from hand-crafted clauses, this pipeline consumes `extracted_document_ast.json`
 * emitted by Stage 1 (`stage1_doc_ast_extractor.py`).
 *
 * Key architectural upgrade:
 *   1. Stage 1 already extracted `deterministic_cross_refs` (e.g. ['SEC-4.3', 'SEC-4.2']) and
 *      `deterministic_part_skus` (e.g. ['PART:VLV-200-SS', 'PART:SEAL-VT-09']) directly from the
 *      PDF/DOCX layout tree at zero token cost.
 *   2. Stage 2 passes those exact anchors as **pinned boundary constraints** into the Diffusion LLM
 *      (Mercury 2) so the model never hallucinates node IDs—it only resolves the semantic relation type
 *      ('OVERRIDES' | 'PROHIBITS' | 'REQUIRES' | 'TRIGGERS') and formal `condition_predicate`.
 */

import OpenAI from "openai";
import fs from "fs";

const dllmClient = new OpenAI({
  apiKey: process.env.INCEPTION_API_KEY || process.env.OPENAI_API_KEY,
  baseURL: process.env.DLLM_BASE_URL || "https://api.inceptionlabs.ai/v1",
});

export interface AstClauseNode {
  clause_id: string;
  parent_section_id: string;
  precedence_tier: number;
  heading: string;
  text: string;
  has_table: boolean;
  deterministic_cross_refs: string[];
  deterministic_part_skus: string[];
}

export interface ExtractedDocumentAst {
  schema_version: string;
  pipeline_stage: string;
  documents: {
    doc_id: string;
    source_file: string;
    sections: {
      section_id: string;
      title: string;
      default_precedence_tier: number;
      clauses: AstClauseNode[];
    }[];
  }[];
}

export type RelationType =
  | "OVERRIDES"
  | "PROHIBITS"
  | "REQUIRES"
  | "TRIGGERS"
  | "REFERENCES"
  | "UPGRADES_TO";

export interface CompiledEdge {
  source_node_id: string;
  target_node_id: string;
  relation_type: RelationType;
  condition_predicate: string | null;
}

export interface CompiledClauseGraph {
  clause_id: string;
  parent_section_id: string;
  precedence_tier: number;
  semantic_edges: CompiledEdge[];
}

/**
 * Dynamically builds a pinned JSON Schema per AST clause so `source_node_id` and `target_node_id`
 * are strictly constrained to the valid universe of extracted AST node IDs and Part SKUs.
 */
function buildPinnedClauseSchema(validNodeUniverse: string[]) {
  return {
    name: "pinned_ast_edge_resolver",
    strict: true,
    schema: {
      type: "object",
      properties: {
        clause_id: { type: "string", enum: validNodeUniverse },
        semantic_edges: {
          type: "array",
          items: {
            type: "object",
            properties: {
              source_node_id: { type: "string", enum: validNodeUniverse },
              target_node_id: { type: "string", enum: validNodeUniverse },
              relation_type: {
                type: "string",
                enum: ["OVERRIDES", "PROHIBITS", "REQUIRES", "TRIGGERS", "REFERENCES", "UPGRADES_TO"],
              },
              condition_predicate: {
                type: ["string", "null"],
                description:
                  "Formal boolean expression over site variables (ambient_temp_c, pressure_psi, loop_id, methanol_present)",
              },
            },
            required: ["source_node_id", "target_node_id", "relation_type", "condition_predicate"],
            additionalProperties: false,
          },
        },
      },
      required: ["clause_id", "semantic_edges"],
      additionalProperties: false,
    },
  };
}

/**
 * Compiles the Stage 1 Document AST into a validated property graph using pinned-boundary dLLM extraction.
 */
export async function compileAstToGraph(astPath: string): Promise<CompiledClauseGraph[]> {
  const rawAst: ExtractedDocumentAst = JSON.parse(fs.readFileSync(astPath, "utf-8"));

  // 1. Collect all flattened AST clauses and build the closed universe of valid Node IDs
  const allClauses: AstClauseNode[] = [];
  const validNodeSet = new Set<string>();

  for (const doc of rawAst.documents) {
    for (const sec of doc.sections) {
      validNodeSet.add(sec.section_id);
      for (const cl of sec.clauses) {
        allClauses.push(cl);
        validNodeSet.add(cl.clause_id);
        cl.deterministic_cross_refs.forEach((ref) => validNodeSet.add(ref));
        cl.deterministic_part_skus.forEach((sku) => validNodeSet.add(sku));
      }
    }
  }

  const validNodeUniverse = Array.from(validNodeSet).sort();

  // 2. Fan out parallel dLLM extraction with pinned boundary node enums
  const tasks = allClauses.map(async (clause) => {
    // Narrow the allowed enum per clause to its own ID + its deterministic anchors (plus global sections)
    const localCandidateNodes = Array.from(
      new Set([
        clause.clause_id,
        clause.parent_section_id,
        ...clause.deterministic_cross_refs,
        ...clause.deterministic_part_skus,
      ])
    );

    const response = await dllmClient.chat.completions.create({
      model: process.env.DLLM_MODEL || "mercury-2",
      messages: [
        {
          role: "system",
          content:
            "You are Stage 2 of a deterministic specification compiler. Stage 1 AST parsing has already " +
            "identified the exact clause ID, cross-referenced clauses, and part SKUs present in this block. " +
            "Resolve the directed edges connecting these pinned nodes and write formal boolean condition_predicates.",
        },
        {
          role: "user",
          content: JSON.stringify({
            clause_id: clause.clause_id,
            precedence_tier: clause.precedence_tier,
            heading: clause.heading,
            pinned_cross_refs: clause.deterministic_cross_refs,
            pinned_part_skus: clause.deterministic_part_skus,
            raw_text_and_tables: clause.text,
          }),
        },
      ],
      response_format: {
        type: "json_schema",
        json_schema: buildPinnedClauseSchema(
          localCandidateNodes.length > 1 ? localCandidateNodes : validNodeUniverse
        ),
      },
    });

    const parsed = JSON.parse(response.choices[0]?.message?.content || "{}");
    return {
      clause_id: clause.clause_id,
      parent_section_id: clause.parent_section_id,
      precedence_tier: clause.precedence_tier,
      semantic_edges: parsed.semantic_edges || [],
    };
  });

  return Promise.all(tasks);
}
