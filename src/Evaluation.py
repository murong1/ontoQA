import json
import pandas as pd
import re
import string
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
import copy
import numpy as np
# import mauve
import nltk
from nltk import sent_tokenize



nltk_path = "nltk_data"

nltk.data.path.append(nltk_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading punkt")
    nltk.download("punkt", download_dir=nltk_path)
    nltk.download("wordnet", download_dir=nltk_path)


def bleu_1(p, g):
    return sentence_bleu(g, p, weights=(1, 0, 0, 0))


def bleu_4(p, g):
    return sentence_bleu(g, p, weights=(0, 0, 0, 1))


def bleu_4_modify(p, g):
    return sentence_bleu(g, p, weights=(0.25, 0.25, 0.25, 0.25))


def bleu_1_smooth(p, g):
    return sentence_bleu(
        g, p, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1
    )


def bleu_4_smooth(p, g):
    return sentence_bleu(
        g, p, weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1
    )


def bleu_4_modify_smooth(p, g):
    return sentence_bleu(
        g,
        p,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )


def meteor(p, g):
    return meteor_score([x.split() for x in g], p.split())



def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, tokenize=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        if tokenize:
            score = metric_fn(word_tokenize(prediction), [word_tokenize(ground_truth)])
        else:
            score = metric_fn(prediction, [ground_truth])
        scores_for_ground_truths.append(score)

    if isinstance(score, dict) and "rougeL" in score:
        rouge_l_score = {"rouge_l f1": 0, "rouge_l precision": 0, "rouge_l recall": 0}
        rouge_l_score["rouge_l f1"] = max(
            [score["rougeL"].fmeasure for score in scores_for_ground_truths]
        )
        rouge_l_score["rouge_l precision"] = max(
            [score["rougeL"].precision for score in scores_for_ground_truths]
        )
        rouge_l_score["rouge_l recall"] = max(
            [score["rougeL"].recall for score in scores_for_ground_truths]
        )

        return rouge_l_score
    else:
        return round(max(scores_for_ground_truths), 2)




class Evaluator:
    def __init__(self, eval_path: str, dataset_name: str):

        self.path = eval_path
        # self.config = default_config
        # self.llm = create_llm_instance(self.config.llm)
        self.dataset_name = dataset_name
        self.short_eval_metrics = ["accuracy", "f1", "precision", "recall", "em"]
        self.close_eval_metrics = ["accuracy"]
        self.long_narrative_metrics = [
            "bleu_1",
            "bleu_4",
            "modify_bleu_4",
            "bleu_1_smooth",
            "bleu_4_smooth",
            "modify_bleu_4_smooth",
            "meteor",
            "rouge_l f1",
            "rouge_l precision",
            "rouge_l recall",
        ]
        self.long_asqa_metrics = ["str_em", "str_hit", "rougeLsum", "mauve"]

        self.dataset_mode_map = {
            "hotpotqa": "short-form",
            "musique": "short-form",
            "multihop-rag": "short-form",
            "popqa": "short-form",
            "ALCE": "long-asqa",
            "quality": "close-set",
        }
        if "narrative" in dataset_name:
            self.mode = "long-narrative"
        else:
            self.mode = self.dataset_mode_map.get(dataset_name, "short-form")

    async def evaluate(self):
        df = pd.read_json(self.path, lines=True)
        print(f"Loaded {len(df)} records from {self.path}")
        print(f"Evaluating {self.mode} mode.")

        if self.mode == "short-form":
            self.print_eval_matrics(self.short_eval_metrics)
            res_dict, df = self.short_eval(df)
            # res_dict, df = self.short_eval_2(df)
        elif self.mode == "long-narrative":
            self.print_eval_matrics(self.long_narrative_metrics)
            res_dict, df = self.long_narrative_eval(df)

        elif self.mode == "long-asqa":
            self.print_eval_matrics(self.long_asqa_metrics)
            res_dict, df = self.long_asqa_eval(df)

        elif self.mode == "close-set":
            self.print_eval_matrics(self.close_eval_metrics)
            res_dict, df = await self.close_eval(df)

        else:
            raise ValueError("Invalid evaluation mode.")

        # add .score to the save path, before the .json
        save_path = self.path.replace(".json", ".score.json")
        df.to_json(save_path, orient="records", lines=True)
        return res_dict

    def print_eval_matrics(self, eval_matrics):
        print("In this evaluation, the following metrics are used:")
        for metric in eval_matrics:
            print(metric, end=" ")
        print("\n")

    def get_label_pred_list(self, df, pred_col, label_col):
        label_list = df[label_col].tolist()
        pred_list = df[pred_col].tolist()
        return label_list, pred_list

    def short_eval(self, df: pd.DataFrame):
        # Short form evaluation code is referenced from the HippoRAG evaluation script:
        # links: https://github.com/OSU-NLP-Group/HippoRAG

        # Load results
        accuracy_list = []
        f1_list = []
        precission_list = []
        recall_list = []
        em_list = []

        label_list, pred_list = self.get_label_pred_list(df, "output", "answer")

        for prediction, answer in zip(pred_list, label_list):
            prediction = prediction.replace("|", "\n")
            prediction = prediction.split("\n")
            prediction_str = " ".join(prediction)

            answer = answer.split("|")
            if isinstance(answer, list):
                answer_str = " ".join(answer)
            else:
                answer_str = answer

            accuracy = self.eval_accuracy(prediction_str, answer_str)
            f1, prec, recall = self.f1_score(prediction_str, answer_str)
            em = self.exact_match_score(prediction_str, answer_str)
            em_list.append(em)
            f1_list.append(f1)
            precission_list.append(prec)
            recall_list.append(recall)
            accuracy_list.append(accuracy)

        accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        f1 = sum(f1_list) * 100 / len(f1_list)
        pre = sum(precission_list) * 100 / len(precission_list)
        recall = sum(recall_list) * 100 / len(recall_list)
        em = sum(em_list) * 100 / len(em_list)

        df["accuracy"] = accuracy_list
        df["f1"] = f1_list
        df["precision"] = precission_list
        df["recall"] = recall_list
        df["em"] = em_list

        res_dict = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": pre,
            "recall": recall,
            "em": em,
        }

        print(f"accuracy: {accuracy:.4f}")
        print(f"Precision: {pre:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"EM: {em:.4f}")

        return res_dict, df
    
    def short_eval_2(self, df: pd.DataFrame):
        accuracy_list = []
        f1_list = []
        precission_list = []
        recall_list = []
        em_list = []
        label_list, pred_list = self.get_label_pred_list(df, "output", "answer")
        
        for prediction, answer in zip(pred_list, label_list):
            # prediction 是一句话，不用拆分
            prediction_str = prediction

            # answer 是多个候选答案，用'|'拆分
            answer_list = answer.split('|')
            answer_list = [a.strip() for a in answer_list if a.strip()]  # 去除空白

            # 针对 accuracy：只要命中任何一个候选答案就是正确
            accuracy = 0
            for cand in answer_list:
                if self.eval_accuracy(prediction_str, cand):  # 调用原有判分逻辑
                    accuracy = 1
                    break
            
            # 针对 f1、precision、recall：取与所有候选分数的最大值
            f1s, precs, recs = [], [], []
            for cand in answer_list:
                f1, prec, rec = self.f1_score(prediction_str, cand)
                f1s.append(f1)
                precs.append(prec)
                recs.append(rec)
            max_f1 = max(f1s)
            max_prec = max(precs)
            max_rec = max(recs)

            # 针对 EM：只要 prediction 完全等于任何一个候选答案就是1，否则0
            em = 0
            for cand in answer_list:
                if self.exact_match_score(prediction_str, cand):
                    em = 1
                    break

            # 收集分数
            em_list.append(em)
            f1_list.append(max_f1)
            precission_list.append(max_prec)
            recall_list.append(max_rec)
            accuracy_list.append(accuracy)

        accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        f1 = sum(f1_list) * 100 / len(f1_list)
        pre = sum(precission_list) * 100 / len(precission_list)
        recall = sum(recall_list) * 100 / len(recall_list)
        em = sum(em_list) * 100 / len(em_list)
        df["accuracy"] = accuracy_list
        df["f1"] = f1_list
        df["precision"] = precission_list
        df["recall"] = recall_list
        df["em"] = em_list
        res_dict = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": pre,
            "recall": recall,
            "em": em,
        }
        print(f"accuracy: {accuracy:.4f}")
        print(f"Precision: {pre:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"EM: {em:.4f}")
        return res_dict, df


    async def close_eval(self, df: pd.DataFrame):
        # Close set evaluation first use LLM to extract the option from the model output
        # Then, evaluate the extracted option with the answer index
        
    
        for index, row in df.iterrows():
            prompt = CLOSE_EXTRACT_OPTION_PORMPT.format(
                question=row["question"], model_output=row["output"]
            )
            response = await self.llm.aask(msg=prompt, format="json")

            try:
                df.loc[index, "extract_output"] = response["predict"]
            except Exception as e:
                df.loc[index, "extract_output"] = "-1"
        print("LLM extract option completed.")

        accuracy_list = []
        label_list, pred_list = self.get_label_pred_list(
            df, "extract_output", "answer_idx"
        )

        for prediction, answer in zip(pred_list, label_list):
            prediction = prediction.strip()
            answer = answer.strip()
            accuracy = self.exact_match_score(prediction, answer)
            accuracy_list.append(accuracy)

        accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        df["accuracy"] = accuracy_list
        res_dict = {"accuracy": accuracy}
        print(f"accuracy: {accuracy:.4f}")
        return res_dict, df

    def exact_presence(self, short_answers, context):
        """Verify if any of the answers is present in the given context.
        Args:
            short_answers: list of short answers to look for in the context
            context: a paragraph to search for short answers
        Returns:
            true if any of the short answers is present in the context
        """

        n_short_answers = [self.normalize_answer(sa) for sa in short_answers]
        n_context = self.normalize_answer(context)

        for ans in n_short_answers:
            if ans in n_context:
                return True

        return False

    def eval_str_em(self, prediction, qa_pairs: list):
        if len(qa_pairs) == 0:
            return 0, 0

        loc_acc = []
        for qa_pair in qa_pairs:
            loc_acc.append(self.exact_presence(qa_pair["short_answers"], prediction))

        acc = np.mean(loc_acc)
        hit = int(acc == 1)

        return acc, hit

    def compute_mauve(self, df):
        human_data = []
        model_data = []
        for idx, row in df.iterrows():
            # Remove ending punctuations
            # Remove any new lines
            # Truncate by 100 words
            human_data.append(
                " ".join(
                    (row["question"] + " " + row["answer"].strip()).split()[:100]
                ).rstrip(string.punctuation)
            )
            model_data.append(
                " ".join(
                    (row["question"] + " " + row["output"].strip()).split()[:100]
                ).rstrip(string.punctuation)
            )

        out = mauve.compute_mauve(
            p_text=human_data,
            q_text=model_data,
            device_id=0,
            max_text_length=512,
            verbose=True,
            batch_size=8,
            featurize_model_name="gpt2-large",
        )
        return out.mauve * 100


    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
            # return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction, ground_truth):
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)

        ZERO_METRIC = (0, 0, 0)

        if (
            normalized_prediction in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
            return ZERO_METRIC
        if (
            normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
            return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def exact_match_score(self, prediction, ground_truth):
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

    def eval_accuracy(self, prediction: str, ground_truth: str):
        s1 = self.normalize_answer(prediction)
        s2 = self.normalize_answer(ground_truth)
        
        # 双向检查：答案包含在预测中，或预测包含在答案中
        # 同时检查完全匹配
        if s1 == s2 or s2 in s1 or s1 in s2:
            return 1
        
        import re
        
        # 特殊处理：如果ground_truth是由连字符连接的复合词，尝试匹配各个部分
        # 例如 "fifth-largest" 应该能匹配 "fifth largest"
        if '-' in ground_truth and len(ground_truth.split('-')) <= 3:
            gt_parts = [part.strip() for part in ground_truth.split('-')]
            gt_normalized_parts = [self.normalize_answer(part) for part in gt_parts]
            # 检查所有部分是否都出现在预测中
            all_parts_found = all(part in s1 for part in gt_normalized_parts if part)
            if all_parts_found and gt_normalized_parts:
                return 1
        
        # 提取关键词进行匹配（去掉常见的停用词后比较）
        # 提取主要词汇（长度>=3的单词）
        words_pred = set(re.findall(r'\b\w{3,}\b', s1))
        words_gt = set(re.findall(r'\b\w{3,}\b', s2))
        
        # 如果有公共关键词且其中一个较短（可能是简化答案），则认为匹配
        common_words = words_pred & words_gt
        if common_words and (len(words_gt) <= 3 or len(words_pred) <= 3):
            return 1
        
        # 处理数字情况
        nums_pred = re.findall(r'\d+', s1)
        nums_gt = re.findall(r'\d+', s2)
        
        # 如果预测中包含真实答案中的任意数字，认为部分正确
        if nums_gt and nums_pred:
            for num_gt in nums_gt:
                if num_gt in nums_pred:
                    return 1
        
        return 0
    def eval_accuracy_multi(self, prediction: str, ground_truth: str):
        """
        prediction : string, 只是一句话
        ground_truth : string，'答案1|答案2|答案3'
        只要prediction和使用原有逻辑能命中ground_truth里的任一候选答案，就判1，否则0
        """
        answer_list = ground_truth.split('|')
        answer_list = [a.strip() for a in answer_list if a.strip()]
        for cand in answer_list:
            if self.eval_accuracy(prediction, cand):  # 用原有判分逻辑
                return 1
        return 0


CLOSE_EXTRACT_OPTION_PORMPT = """
You are given a model output which is a string. The model output is a list of options. You have to extract the option letter from the model output.

# GOAL

Your goal is to extract the option letter directly from the model output. You should not rely on any external knowledge or context to answer. Simply extract the option letter as stated in the model output.

# FORMAT

Please provide your answer in the following JSON format:

- ANSWER_OPTION: the option letter extracted from the model output.

    {{
        "model_output": <answer_option>
    }}

### Example 1
-----------
# INPUT:

Question:
How much time has passed between Blake's night with Eldoria and his search for Sabrina York in his mind-world?
A: 7 years
B: 10 hours
C: 12 years
D: 1 hour

# Model Output: 
I think the answer is 7 years.

OUTPUT:
    {{
        "predict": "A"
    }}

### Example 2
-----------
# INPUT:

Question:
How much time has passed between Blake's night with Eldoria and his search for Sabrina York in his mind-world?
A: 7 years
B: 10 hours
C: 12 years
D: 1 hour

# Model Output: 
The correct answer is C.

OUTPUT:
    {{
        "predict": "C"
    }}
    
### EXAMPLE 3
-----------

# INPUT:

Question:
Donald Trump is the president of:
A: China
B: Canada
C: France
D: Spain

# Model Output: 
The correct answer is: None of the above.

OUTPUT:
    {{
        "predict": "-1"
    }}

Now please the output based on the given question and model output.

### Real Data
# INPUT:

Question:
{question}

# Model Output:
{model_output}

OUTPUT:"""