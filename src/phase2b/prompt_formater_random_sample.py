class PromptFormatter:
    """Create consistent prompts across all models - WITH CAUSAL INTERVENTION"""

    @staticmethod
    def format(question: str, task_type: str = "math", condition: str = "control", passage: str = None) -> str:
        """
        Format question with explicit instructions for step-by-step reasoning.
        
        Args:
            question: The question text
            task_type: Type of task (math, reasoning, commonsense)
            condition: Intervention condition (control, brief, direct, length_only)
        """

        # ================================================================
        # CAUSAL INTERVENTION: Four conditions
        # control     → original prompt, unconstrained generation
        # brief       → brevity instruction in prompt + short token limit
        # direct      → answer-only instruction in prompt
        # length_only → SAME prompt as control, token cap at generation level only
        #               This isolates token budget from prompt framing effect
        # ================================================================
        
        if task_type == "math":
            if condition == "control":
                return f"""Problem: {question}

Solution:"""
            
            elif condition == "brief":
                return f"""Problem: {question}

Provide a BRIEF solution in under 50 words. Show only the essential calculation steps.

Solution:"""
            
            elif condition == "direct":
                return f"""Problem: {question}

Provide ONLY the final numerical answer. No explanation or reasoning.

Answer:"""

            elif condition == "length_only":
                # IDENTICAL to control — no brevity instruction
                # Token limit enforced in probe_sample() at generation level
                return f"""Problem: {question}

Solution:"""

        elif task_type == "reasoning":
            if condition == "control":
                return f"""Read the passage carefully and answer the question.

Passage: {passage}

Question: {question}

Think carefully about what the passage says. Answer with only "Yes" or "No".

Answer:"""
            
            elif condition == "brief":
                return f"""Read the passage and answer.

Passage: {passage}

Question: {question}

Answer in 10 words or less: Yes or No, and why.

Answer:"""
            
            elif condition == "direct":
                return f"""Read the passage and answer.

Passage: {passage}

Question: {question}

Answer ONLY: Yes or No

Answer:"""

            elif condition == "length_only":
                # IDENTICAL to control — no brevity instruction
                return f"""Read the passage carefully and answer the question.

Passage: {passage}

Question: {question}

Think carefully about what the passage says. Answer with only "Yes" or "No".

Answer:"""

        elif task_type == "commonsense":
            if condition == "control":
                return f"""Question: {question}

        Answer:"""
            
            elif condition == "brief":
                return f"""Answer this multiple choice question.

{question}

Answer with just the letter and ONE sentence explanation.

Answer:"""
            
            elif condition == "direct":
                return f"""Answer this multiple choice question.

{question}

Answer with ONLY the letter (A, B, C, D, or E).

Answer:"""

            elif condition == "length_only":
                # IDENTICAL to control — no brevity instruction
                return f"""Question: {question}

        Answer:"""

        else:
            return question