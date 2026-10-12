// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "swift-generate",
    platforms: [.macOS("14.0")],
    products: [
        .executable(name: "MLXGenerate", targets: ["MLXGenerate"]),
    ],
    dependencies: [
        .package(path: "../mlx-swift-lm"),
        // Same pairing as d-inference 7796fe2 (libs/mlx-swift 6923a80 + libs/mlx-swift-lm 3fd4944):
        // a root path dependency overrides mlx-swift-lm's URL pin, as provider-swift does.
        .package(path: "../mlx-swift"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.2"),
    ],
    targets: [
        .executableTarget(
            name: "MLXGenerate",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXGenerate"
        ),
    ]
)
