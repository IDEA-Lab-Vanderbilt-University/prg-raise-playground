import { BlockMetadata, Argument, ReturnTypeByBlockType, ScratchBlockType, ToArguments, AnyBlock } from "$common/types";
import { block } from "$common/extension/decorators/blocks";
import { ExtensionInstance } from "..";
import { TypedMethodDecorator } from ".";
import type BlockUtilityWithID from "$scratch-vm/engine/block-utility";


const process = (type: ScratchBlockType, strings: TemplateStringsArray, ...args: any[]) => {
    if (args.length === 0) return { type, text: strings[0], };
    const text = (...placeholders: any[]) => strings.map((str, i) => `${str}${placeholders[i] ?? ""}`).join("");
    if (args.length === 1) return { type, text, arg: args[0] };
    return { type, text, args };
}

export function makeDecorator<T extends ScratchBlockType>(type: T): TemplateEngine<T>["execute"] {
    // function takes T and returns a function of TemplateEngine type
    // TemplateEngine returns based on the ScratchType of the block
    return function decoratorFn(builderOrStrings: Parameters<TemplateEngine<T>["execute"]>[0], ...args) {
        return function (target, context) {

            type BlockMetadataFunction = (instance: ExtensionInstance) => BlockMetadata<(...args: any[]) => any>;
            type StaticBlockMetadata = BlockMetadata<(...args: any[]) => any>;

            const input = typeof builderOrStrings == "function"
                ? (instance: ExtensionInstance) => (builderOrStrings as any)(instance, process.bind(null, type)) satisfies BlockMetadataFunction
                : process(type, builderOrStrings, ...args) as StaticBlockMetadata;

            return block(input)(target, context);
        }
    }
}

namespace Utility {
    export type TaggedTemplate<Args extends any[], Return> = (strings: TemplateStringsArray, ...args: Args) => Return;
}

namespace Argument {
    type TRemoveUtil<T extends any[]> = T extends [...infer R extends any[], BlockUtilityWithID] ? R : T;
    // Maya note: thought ToArguments was the equivalent, but TypeScript does not like it....
    export type MapToScratch<T extends any[], Internal extends TRemoveUtil<T> = TRemoveUtil<T>> = {
        [k in keyof Internal]: Argument<Internal[k]>
    }
}

// TODO: Restrict return based on Scratch type
interface TemplateEngine<TBlockType extends ScratchBlockType> {
    /**
     * 
     */
    execute<
        const This extends ExtensionInstance,
        const Args extends any[],
        const Return extends ReturnTypeByBlockType<TBlockType>
    >
        (
            strings: TemplateStringsArray, ...args: Argument.MapToScratch<Args>
        ): TypedMethodDecorator<This, Args, Return, ((...Args) => Return)>;

    /**
     * 
     */
    execute<
        const This extends ExtensionInstance,
        const Args extends any[],
        const Return extends ReturnTypeByBlockType<TBlockType>,
    >
        (
            builder: (
                instance: This,
                tag: Utility.TaggedTemplate<Argument.MapToScratch<Args>, BlockMetadata<(...args: Args) => Return>>
            ) => BlockMetadata<(...args: Args) => Return>
        ): TypedMethodDecorator<This, Args, Return, ((...Args) => Return)>;
}


export const scratch = {
    reporter: makeDecorator("reporter"),
    command: makeDecorator("command"),
}