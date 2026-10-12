// swift-tools-version: 6.1
// Smoke test: one tensor op on the GPU through mlx-swift (Layr-Labs fork, CUDA backend on Linux).
import PackageDescription

let package = Package(
    name: "MLXSmoke",
    dependencies: [.package(path: "../mlx-swift")],
    targets: [
        .executableTarget(
            name: "MLXSmoke",
            dependencies: [.product(name: "MLX", package: "mlx-swift")]
        )
    ]
)
