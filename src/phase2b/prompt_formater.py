class PromptFormatter:
    """Create consistent prompts across all models - WITH CAUSAL INTERVENTION"""

    @staticmethod
    def format(question: str, task_type: str = "math", condition: str = "control", passage: str = None) -> str:
        """
        Format question with explicit instructions for step-by-step reasoning.
        
        Args:
            question: The question text
            task_type: Type of task (math, reasoning, commonsense)
            condition: Intervention condition (control, brief, direct)
        """

        # ================================================================
        # CAUSAL INTERVENTION: Three conditions
        # ================================================================
        
        if task_type == "math":
            if condition == "control":
                # Original prompt - allows full reasoning
                return f"""Problem: {question}

Solution:"""
            
            elif condition == "brief":
                # INTERVENTION: Force brevity
                return f"""Problem: {question}

Provide a BRIEF solution in under 50 words. Show only the essential calculation steps.

Solution:"""
            
            elif condition == "direct":
                # INTERVENTION: Force direct answer only
                return f"""Problem: {question}

Provide ONLY the final numerical answer. No explanation or reasoning.

Answer:"""

        elif task_type == "reasoning":
            if condition == "control":
                # Original prompt
                return f"""Read the passage carefully and answer the question.

                Passage: {passage}

                Question: {question}

                Think carefully about what the passage says. Answer with only "Yes" or "No".

                Answer:"""
            
            elif condition == "brief":
                # INTERVENTION: Force brevity
                return f"""Read the passage and answer.

Passage: {passage}

Question: {question}

Answer in 10 words or less: Yes or No, and why.

Answer:"""
            
            elif condition == "direct":
                # INTERVENTION: Direct answer only
                return f"""Read the passage and answer.

Passage: {passage}

Question: {question}

Answer ONLY: Yes or No

Answer:"""

        elif task_type == "commonsense":
            if condition == "control":
                return f"""Question: {question}

        Answer:"""
            
            elif condition == "brief":
                # INTERVENTION: Force brevity
                return f"""Answer this multiple choice question.

{question}

Answer with just the letter and ONE sentence explanation.

Answer:"""
            
            elif condition == "direct":
                # INTERVENTION: Direct answer only
                return f"""Answer this multiple choice question.

{question}

Answer with ONLY the letter (A, B, C, D, or E).

Answer:"""

        else:
            return question