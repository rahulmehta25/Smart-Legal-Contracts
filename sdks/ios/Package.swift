// swift-tools-version: 5.9
// ArbitrationSDK - iOS SDK for arbitration clause detection and analysis

import PackageDescription

let package = Package(
    name: "ArbitrationSDK",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .tvOS(.v15),
        .watchOS(.v8)
    ],
    products: [
        .library(
            name: "ArbitrationSDK",
            targets: ["ArbitrationSDK"]
        ),
        .library(
            name: "ArbitrationUI",
            targets: ["ArbitrationUI"]
        ),
        .library(
            name: "ArbitrationAR",
            targets: ["ArbitrationAR"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.8.0"),
        .package(url: "https://github.com/realm/realm-swift.git", from: "10.45.0"),
        .package(url: "https://github.com/apple/swift-crypto.git", from: "3.0.0"),
        .package(url: "https://github.com/swiftlang/swift-testing.git", from: "0.7.0")
    ],
    targets: [
        // Core SDK
        .target(
            name: "ArbitrationSDK",
            dependencies: [
                "Alamofire",
                .product(name: "RealmSwift", package: "realm-swift"),
                .product(name: "Crypto", package: "swift-crypto")
            ],
            path: "Sources/ArbitrationSDK",
            resources: [
                .process("Resources")
            ]
        ),
        
        // UI Components
        .target(
            name: "ArbitrationUI",
            dependencies: ["ArbitrationSDK"],
            path: "Sources/ArbitrationUI",
            resources: [
                .process("Resources")
            ]
        ),
        
        // AR Capabilities
        .target(
            name: "ArbitrationAR",
            dependencies: ["ArbitrationSDK"],
            path: "Sources/ArbitrationAR"
        ),
        
        // Tests
        .testTarget(
            name: "ArbitrationSDKTests",
            dependencies: [
                "ArbitrationSDK",
                "ArbitrationUI",
                "ArbitrationAR",
                .product(name: "Testing", package: "swift-testing")
            ],
            path: "Tests"
        )
    ]
)