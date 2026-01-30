//
//  Generate.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 27.09.2025.
//

import Foundation
import MLXLMCommon
import MLX

#if canImport(FoundationModels)
import FoundationModels
#endif

public func generate(
    input: LMInput,
    parameters: GenerateParameters = GenerateParameters(),
    context: ModelContext,
    grammar: Grammar,
    didGenerate: ([Int]) -> GenerateDisposition = { _ in .more }
) async throws -> GenerateResult {
    let sampler = parameters.sampler()
    let processor = try await GrammarMaskedLogitProcessor.from(configuration: context.configuration, grammar: grammar)
    let iterator = try TokenIterator(input: input, model: context.model, processor: processor, sampler: sampler)
    let result = generate(input: input, context: context, iterator: iterator, didGenerate: didGenerate)
    return result
}
