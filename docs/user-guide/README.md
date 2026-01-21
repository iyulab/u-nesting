# U-Nesting User Guide

Welcome to the U-Nesting User Guide. This documentation covers installation, API usage, and algorithm details for the U-Nesting spatial optimization engine.

## Table of Contents

1. [Getting Started](./getting-started.md)
2. [2D Nesting Guide](./nesting-2d.md)
3. [3D Bin Packing Guide](./packing-3d.md)
4. [Algorithm Reference](./algorithms.md)
5. [FFI Integration](./ffi-integration.md)
6. [Performance Tuning](./performance.md)

## Quick Links

- [Python Package (PyPI)](https://pypi.org/project/u-nesting/)
- [C# Package (NuGet)](https://www.nuget.org/packages/UNesting/)
- [GitHub Repository](https://github.com/iyulab/U-Nesting)

## What is U-Nesting?

U-Nesting is a domain-agnostic spatial optimization engine that provides:

- **2D Nesting**: Optimal placement of irregular polygons on sheets (cutting stock problem)
- **3D Bin Packing**: Optimal placement of boxes in containers

Unlike domain-specific tools, U-Nesting is designed as a pure computation engine. It handles the spatial optimization mathematics while leaving domain-specific decisions (material properties, industry constraints, etc.) to the consuming application.

## Supported Platforms

- **Native**: Windows (x64), Linux (x64, ARM64), macOS (x64, ARM64)
- **Python**: 3.8+ via PyO3 bindings
- **C#/.NET**: .NET 6.0+ via P/Invoke
- **C/C++**: Direct FFI via C headers
