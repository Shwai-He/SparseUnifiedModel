from typing import List

from tokenizers import Tokenizer


class S3BpeTokenizer():
    def __init__(self, bpe_model, mapping_file) -> None:
        # s3 bpe model
        self.sp = Tokenizer.from_file(bpe_model)
        # s3 token转为中文token
        self.mapping_file = mapping_file
        self.s3_to_zh = self.get_s3_to_zh()
        self.zh_to_s3 = {v: k for k, v in self.s3_to_zh.items()}

    def get_s3_to_zh(self):
        s3_to_zh = {}
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split('\t')
                assert len(arr) == 2
                s3_to_zh[int(arr[0])] = arr[1]
        return s3_to_zh

    def encode(self, s3_code: List[int]):
        """
        s3_code: List[int], eg: [1, 2, 3, 4]
        return: List[int], List[str], eg: [3, 2, 3, 8] ['我', '是', '我', '的']
        """
        new_s3_code = [self.s3_to_zh[x] for x in s3_code]
        new_s3_code = ''.join(new_s3_code)
        output = self.sp.encode(new_s3_code)
        return output.ids, output.tokens

    def decode(self, s3_bpe_token: List[int]):
        """
        s3_bpe_token: s3经过bpe编码之后的id [1, 2, 3, 4]
        return: 原始s3 token [4, 5, 7, 8]
        """
        s3_zh_str = self.sp.decode(s3_bpe_token)
        s3_zh_str = s3_zh_str.replace(' ', '')
        return [self.zh_to_s3[x] for x in s3_zh_str]

