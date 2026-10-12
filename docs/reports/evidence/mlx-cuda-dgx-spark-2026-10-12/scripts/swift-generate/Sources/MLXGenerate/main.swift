import Foundation
import MLX
import MLXHuggingFace
import Tokenizers
import MLXLLM
import MLXLMCommon

// Prompt encoding: `tokenizer.encode(text:)` → `encode(text:addSpecialTokens: true)` (Tokenizer.swift),
// matching Python mlx_trace.py `tokenizer.encode(p["text"])` (HF default add_special_tokens=True).
// Generation: `TokenIterator` stops only at maxTokens, not EOS (Evaluate.swift `next()`).

struct PromptRow: Decodable {
    let id: String
    let text: String
}

struct Args {
    var model = ""
    var prompts = ""
    var maxTokens = 64
    var out = ""
    var label = ""
}

func parseArgs() throws -> Args {
    var a = Args()
    var it = Array(CommandLine.arguments.dropFirst()).makeIterator()
    while let flag = it.next() {
        switch flag {
        case "--model": a.model = try needValue(&it, flag)
        case "--prompts": a.prompts = try needValue(&it, flag)
        case "--max-tokens": a.maxTokens = Int(try needValue(&it, flag)) ?? 64
        case "--out": a.out = try needValue(&it, flag)
        case "--label": a.label = try needValue(&it, flag)
        default:
            throw NSError(domain: "MLXGenerate", code: 1, userInfo: [NSLocalizedDescriptionKey: "unknown arg: \(flag)"])
        }
    }
    guard !a.model.isEmpty, !a.prompts.isEmpty, !a.out.isEmpty else {
        throw NSError(domain: "MLXGenerate", code: 1, userInfo: [NSLocalizedDescriptionKey:
            "usage: MLXGenerate --model DIR --prompts FILE --out FILE [--max-tokens N] [--label STR]"])
    }
    return a
}

func needValue(_ it: inout IndexingIterator<[String]>, _ flag: String) throws -> String {
    guard let v = it.next() else {
        throw NSError(domain: "MLXGenerate", code: 1, userInfo: [NSLocalizedDescriptionKey: "missing value for \(flag)"])
    }
    return v
}

func readPrompts(path: String) throws -> [PromptRow] {
    let url = URL(fileURLWithPath: path)
    let data = try String(contentsOf: url, encoding: .utf8)
    var rows: [PromptRow] = []
    for (i, line) in data.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
        let s = line.trimmingCharacters(in: .whitespaces)
        if s.isEmpty { continue }
        guard let d = s.data(using: .utf8) else { continue }
        do {
            rows.append(try JSONDecoder().decode(PromptRow.self, from: d))
        } catch {
            throw NSError(domain: "MLXGenerate", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "\(path):\(i + 1): expected {\"id\",\"text\"}"])
        }
    }
    guard !rows.isEmpty else {
        throw NSError(domain: "MLXGenerate", code: 1, userInfo: [NSLocalizedDescriptionKey: "no prompts in \(path)"])
    }
    return rows
}

func jsonLine(_ obj: [String: Any]) throws -> String {
    let data = try JSONSerialization.data(withJSONObject: obj, options: [])
    guard let s = String(data: data, encoding: .utf8) else { throw NSError(domain: "MLXGenerate", code: 1) }
    return s
}

@main
struct MLXGenerate {
    static func main() async {
        do {
            try await run()
            // Linux/CUDA: skip static teardown that can abort after successful output.
            Foundation.exit(0)
        } catch {
            FileHandle.standardError.write(Data("\(error.localizedDescription)\n".utf8))
            Foundation.exit(1)
        }
    }

    static func run() async throws {
        let args = try parseArgs()
        let device = Device.defaultDevice()
        print(device)

        let modelURL = URL(fileURLWithPath: args.model, isDirectory: true).standardizedFileURL
        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: #huggingFaceTokenizerLoader()
        )

        let prompts = try readPrompts(path: args.prompts)
        let outURL = URL(fileURLWithPath: args.out)
        FileManager.default.createFile(atPath: outURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: outURL)

        let env: [String: Any] = [
            "type": "env",
            "label": args.label,
            "platform": "linux-swift",
            "model_path": modelURL.path,
            "device": String(describing: device),
        ]
        try handle.write(contentsOf: Data((try jsonLine(env) + "\n").utf8))

        let params = GenerateParameters(maxTokens: args.maxTokens, temperature: 0)
        for row in prompts {
            let t0 = ContinuousClock.now
            let result = try await container.perform { context in
                let promptIds = context.tokenizer.encode(text: row.text)
                let input = LMInput(tokens: MLXArray(promptIds.map { Int32($0) }))
                var iterator = try TokenIterator(input: input, model: context.model, parameters: params)
                var tokens: [Int] = []
                while tokens.count < args.maxTokens, let tok = iterator.next() {
                    tokens.append(tok)
                }
                let text = context.tokenizer.decode(tokenIds: tokens)
                return (promptIds, tokens, text)
            }
            let elapsed = ContinuousClock.now - t0
            let rec: [String: Any] = [
                "id": row.id,
                "prompt_ids": result.0,
                "tokens": result.1,
                "logprobs": NSNull(),
                "margins": NSNull(),
                "text": result.2,
            ]
            try handle.write(contentsOf: Data((try jsonLine(rec) + "\n").utf8))
            let secs = Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            print("\(row.id): \(result.1.count) tokens, \(String(format: "%.3f", secs))s")
        }
    }
}
