from typing import List

def decoder_parser(outputs: List[str], formatted_prompts: List[str], prepoc: callable):
    """Removes the prompt from the text and calls `prepoc` on the completion

    Args:
        outputs (List[str]): model outputs
        formatted_prompts (List[str]): formatted prompts
        prepoc (callable): prepoc function

    Returns:
        List: processed outputs
    """
    ret_val = []
    for output, formatted_prompt in zip(outputs, formatted_prompts):
        ret_val.append(prepoc(output[len(formatted_prompt) :]))
    return ret_val