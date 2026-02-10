import os
import re
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import json

class BrazilianPlatePatterns:
    """
    Brazilian plate patterns based on wikipedia:
    https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Brazil
    """
    def __init__(self):
        # State digit_mapping.
        self.state_digit_mapping = {
            '1': 'DF, GO, MT, MS, TO',
            '2': 'AC, AP, AM, PA, RO, RR',
            '3': 'CE, MA, PI',
            '4': 'AL, PB, PE, RN',
            '5': 'BA, SE',
            '6': 'MG',
            '7': 'ES, RJ',
            '8': 'SP',
            '9': 'PR, SC',
            '0': 'RS'
        }
        
        # State abbreviations mapping.
        self.state_abbr = {
            'PR': 'Paraná',
            'SP': 'São Paulo',
            'MG': 'Minas Gerais',
            'MA': 'Maranhão',
            'MS': 'Mato Grosso do Sul',
            'CE': 'Ceará',
            'SE': 'Sergipe',
            'RS': 'Rio Grande do Sul',
            'DF': 'Distrito Federal',
            'BA': 'Bahia',
            'PA': 'Pará',
            'AM': 'Amazonas',
            'MT': 'Mato Grosso',
            'GO': 'Goiás',
            'PE': 'Pernambuco',
            'RJ': 'Rio de Janeiro',
            'PI': 'Piauí',
            'SC': 'Santa Catarina',
            'PB': 'Paraíba',
            'ES': 'Espírito Santo',
            'AL': 'Alagoas',
            'TO': 'Tocantins',
            'RN': 'Rio Grande do Norte',
            'AC': 'Acre',
            'RR': 'Roraima',
            'RO': 'Rondônia',
            'AP': 'Amapá'
        }
        
        self.state_sequences = self._parse_state_sequences()
        
        self.brazilian_patterns = {
            'old_system': {
                'format': r'^[A-Z]{3}\d{4}$',  # AAA-9999 format.
                'valid_letters': set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                'valid_digits': set('0123456789'),
                'state_codes': self.state_digit_mapping,
                'special_sequences': {
                    'AAA': 'Diplomatic corps',
                    'CDD': 'Consular corps',
                    'CMD': 'Military command',
                    'MUN': 'Municipal government',
                    'GOV': 'State government',
                    'PTR': 'Trailers',
                }
            },
            'mercosur': {
                'format': r'^[A-Z]{3}\d[A-Z]\d{2}$',  # AAA9A99 format.
                'valid_letters': set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                'valid_digits': set('0123456789'),
            }
        }
        
        # Confusion matrix.
        self.confusion_matrix = {
            '0': ['O', 'D', 'Q', '8'],
            '1': ['I', 'L', '7', 'T'],
            '2': ['Z', 'S', '7'],
            '3': ['8', 'B', '5'],
            '4': ['A', 'H', '9'],
            '5': ['S', '6', '3'],
            '6': ['G', '8', '5', '0'],
            '7': ['1', 'T', '2'],
            '8': ['B', '3', '6', '0'],
            '9': ['G', '4', '7'],
            'A': ['4', 'H', 'R', 'N'],
            'B': ['8', '3', 'R', 'D'],
            'C': ['G', 'O', 'Q'],
            'D': ['0', 'O', 'B'],
            'E': ['F', 'B', '3'],
            'F': ['E', 'P', 'R'],
            'G': ['6', 'C', '9', 'Q'],
            'H': ['4', 'A', 'N', 'M'],
            'I': ['1', 'L', 'T', '7'],
            'J': ['T', 'I', '1'],
            'K': ['X', 'R', 'H'],
            'L': ['1', 'I', '7', 'T'],
            'M': ['N', 'H', 'W'],
            'N': ['M', 'H', 'A'],
            'O': ['0', 'D', 'Q', 'C'],
            'P': ['F', 'R', 'B'],
            'Q': ['0', 'O', 'G', 'C'],
            'R': ['P', 'B', 'K', 'A'],
            'S': ['5', '2', '8'],
            'T': ['7', 'I', 'J', '1'],
            'U': ['V', 'W', '0'],
            'V': ['U', 'W', 'Y'],
            'W': ['M', 'V', 'U'],
            'X': ['K', 'Y', 'H'],
            'Y': ['V', 'X', '7'],
            'Z': ['2', '7', '5']
        }
    
    def _parse_state_sequences(self) -> Dict[str, List[Tuple[str, str]]]:
        """State sequence table from the wikipedia page - COMPLETE VERSION"""
        state_sequences = {}
        
        # Paraná (PR)
        state_sequences['PR'] = [
            ('AAA', 'BEZ'), ('RHA', 'RHZ'), ('SDP', 'SFO'), 
            ('TAI', 'TBZ'), ('UAS', 'UCZ')
        ]
        
        # São Paulo (SP)
        state_sequences['SP'] = [
            ('BFA', 'GKI'), ('SAV0A01', 'SAV1A00'), ('QSN', 'QSZ'),
            ('SSR', 'SWZ'), ('TIO', 'TMJ'), ('UDA', 'UGV')
        ]
        
        # Minas Gerais (MG)
        state_sequences['MG'] = [
            ('GKJ', 'HOK'), ('NXX', 'NYG'), ('OLO', 'OMH'),
            ('OOV', 'ORC'), ('OWH', 'OXK'), ('PUA', 'PZZ'),
            ('QMQ', 'QQZ'), ('QUA', 'QUZ'), ('QWR', 'QXZ'),
            ('RFA', 'RGD'), ('RMD', 'RNZ'), ('RTA', 'RVZ'),
            ('SHB', 'SJI'), ('SYA', 'SYZ'), ('UAI', 'UAI'),
            ('TCA', 'TEZ'), ('TWY', 'UAH')
        ]
        
        # Maranhão (MA)
        state_sequences['MA'] = [
            ('HOL', 'HQE'), ('NHA', 'NHT'), ('NMP', 'NNI'),
            ('NWS', 'NXQ'), ('OIR', 'OJQ'), ('OXQ', 'OXZ'),
            ('PSA', 'PTZ'), ('ROA', 'ROZ'), ('SMM', 'SNJ'),
            ('UJM', 'UJZ')
        ]
        
        # Mato Grosso do Sul (MS)
        state_sequences['MS'] = [
            ('HQF', 'HTW'), ('NRF', 'NSD'), ('OOG', 'OOU'),
            ('QAA', 'QAZ'), ('REW', 'REZ'), ('RWA', 'RWJ'),
            ('SLW', 'SML')
        ]
        
        # Ceará (CE)
        state_sequences['CE'] = [
            ('HTX', 'HZA'), ('NQL', 'NRE'), ('NUM', 'NVF'),
            ('OCB', 'OCU'), ('OHX', 'OIQ'), ('ORN', 'OSV'),
            ('OZA', 'OZA'), ('PMA', 'POZ'), ('RIA', 'RIL'),
            ('SAN', 'SAU'), ('SAV1A01', 'SBV'), ('THN', 'TIN')
        ]
        
        # Sergipe (SE)
        state_sequences['SE'] = [
            ('HZB', 'IAP'), ('NVG', 'NVN'), ('OEJ', 'OES'),
            ('OZB', 'OZB'), ('QKN', 'QKZ'), ('QMA', 'QMP'),
            ('RQW', 'RRH'), ('TNU', 'TOD')
        ]
        
        # Rio Grande do Sul (RS)
        state_sequences['RS'] = [
            ('IAQ', 'JDO'), ('TQO', 'TRW')
        ]
        
        # Distrito Federal (DF)
        state_sequences['DF'] = [
            ('JDP', 'JKR'), ('OVM', 'OVV'), ('OZW', 'PBZ'),
            ('REC', 'REV'), ('SGN', 'SGZ'), ('SSF', 'SSQ'),
            ('TUY', 'TUZ'), ('TVK', 'TVK'), ('UIV', 'UJL')
        ]
        
        # Bahia (BA)
        state_sequences['BA'] = [
            ('JKS', 'JSZ'), ('NTD', 'NTW'), ('NYH', 'NZZ'),
            ('OKI', 'OLG'), ('OUF', 'OVD'), ('OZC', 'OZV'),
            ('PJA', 'PLZ'), ('QTU', 'QTZ'), ('RCO', 'RDR'),
            ('RPA', 'RPZ'), ('SJJ', 'SKT'), ('TGR', 'THH'),
            ('TMK', 'TNG')
        ]
        
        # Pará (PA)
        state_sequences['PA'] = [
            ('JTA', 'JWE'), ('NSE', 'NTC'), ('OBT', 'OCA'),
            ('OFI', 'OFW'), ('OSW', 'OTZ'), ('QDA', 'QEZ'),
            ('QVA', 'QVZ'), ('RWK', 'RXJ'), ('SZA', 'SZZ'),
            ('TVL', 'TWK')
        ]
        
        # Amazonas (AM)
        state_sequences['AM'] = [
            ('JWF', 'JXY'), ('NOI', 'NPB'), ('OAA', 'OAO'),
            ('OXM', 'OXM'), ('PHA', 'PHZ'), ('QZA', 'QZZ'),
            ('TAA', 'TAH'), ('TRX', 'TSO')
        ]
        
        # Mato Grosso (MT)
        state_sequences['MT'] = [
            ('JXZ', 'KAU'), ('NIY', 'NJW'), ('NPC', 'NPQ'),
            ('NTX', 'NUG'), ('OAP', 'OBS'), ('QBA', 'QCZ'),
            ('RAK', 'RAZ'), ('RRI', 'RRZ'), ('SPC', 'SQP')
        ]
        
        # Goiás (GO)
        state_sequences['GO'] = [
            ('KAV', 'KFC'), ('NFC', 'NGZ'), ('NJX', 'NLU'),
            ('NVO', 'NWR'), ('OGH', 'OHA'), ('OMI', 'OOF'),
            ('PQA', 'PRZ'), ('QTN', 'QTS'), ('RBK', 'RCN'),
            ('SBW', 'SDO'), ('TFA', 'TGN')
        ]
        
        # Pernambuco (PE)
        state_sequences['PE'] = [
            ('KFD', 'KME'), ('NXU', 'NXW'), ('PEE', 'PFQ'),
            ('PFR', 'PGK'), ('PGL', 'PGU'), ('OYL', 'OYZ'),
            ('PCA', 'PED'), ('PGV', 'PGZ'), ('QYA', 'QYZ'),
            ('RZE', 'RZZ'), ('SNK', 'SPB'), ('UHJ', 'UII')
        ]
        
        # Rio de Janeiro (RJ)
        state_sequences['RJ'] = [
            ('KMF', 'LVE'), ('RIO', 'RIO'), ('RIP', 'RKV'),
            ('SQV', 'SSE'), ('TTA', 'TUX')
        ]
        
        # Piauí (PI)
        state_sequences['PI'] = [
            ('LVF', 'LWQ'), ('NHU', 'NIX'), ('ODU', 'OEI'),
            ('OUA', 'OUE'), ('OVW', 'OVY'), ('PIA', 'PIZ'),
            ('QRN', 'QRZ'), ('RSG', 'RST'), ('SLM', 'SLV'),
            ('UKF', 'UKN')
        ]
        
        # Santa Catarina (SC)
        state_sequences['SC'] = [
            ('LWR', 'MMM'), ('OKD', 'OKH'), ('QHA', 'QJZ'),
            ('QTK', 'QTM'), ('RAA', 'RAJ'), ('RDS', 'REB'),
            ('RKW', 'RLP'), ('RXK', 'RYZ'), ('SXA', 'SXZ'),
            ('TPI', 'TQE')
        ]
        
        # Paraíba (PB)
        state_sequences['PB'] = [
            ('MMN', 'MOW'), ('NPR', 'NQK'), ('OET', 'OFH'),
            ('OFX', 'OGG'), ('OXO', 'OXO'), ('QFA', 'QFZ'),
            ('QSA', 'QSM'), ('RLQ', 'RLZ'), ('SKU', 'SLF'),
            ('TOT', 'TPH')
        ]
        
        # Espírito Santo (ES)
        state_sequences['ES'] = [
            ('MOX', 'MTZ'), ('OCV', 'ODT'), ('OVE', 'OVF'),
            ('OVH', 'OVL'), ('OYD', 'OYK'), ('PPA', 'PPZ'),
            ('QRB', 'QRM'), ('RBA', 'RBJ'), ('RQM', 'RQV'),
            ('SFP', 'SGM'), ('TOE', 'TOS')
        ]
        
        # Alagoas (AL)
        state_sequences['AL'] = [
            ('MUA', 'MVK'), ('NLV', 'NMO'), ('OHB', 'OHK'),
            ('ORD', 'ORM'), ('OXN', 'OXN'), ('QLA', 'QLM'),
            ('QTT', 'QTT'), ('QWG', 'QWL'), ('RGO', 'RGU'),
            ('SAA', 'SAJ'), ('RGV', 'RGZ'), ('TNH', 'TNT')
        ]
        
        # Tocantins (TO)
        state_sequences['TO'] = [
            ('MVL', 'MXG'), ('OLH', 'OLN'), ('OYA', 'OYC'),
            ('QKA', 'QKM'), ('QWA', 'QWF'), ('RSA', 'RSF'),
            ('RIM', 'RIN'), ('RMA', 'RMC'), ('TVA', 'TVD')
        ]
        
        # Rio Grande do Norte (RN)
        state_sequences['RN'] = [
            ('MXH', 'MZM'), ('NNJ', 'NOH'), ('OJR', 'OKC'),
            ('OVZ', 'OWG'), ('QGA', 'QGZ'), ('RGN', 'RGN'),
            ('RGE', 'RGM'), ('RQA', 'RQL'), ('TSP', 'TSZ')
        ]
        
        # Acre (AC)
        state_sequences['AC'] = [
            ('MZN', 'NAG'), ('NXR', 'NXT'), ('OVG', 'OVG'),
            ('OXP', 'OXP'), ('QLU', 'QLZ'), ('QWM', 'QWQ'),
            ('SHA', 'SHA'), ('SQQ', 'SQU')
        ]
        
        # Roraima (RR)
        state_sequences['RR'] = [
            ('NAH', 'NBA'), ('NUH', 'NUL'), ('RZA', 'RZD')
        ]
        
        # Rondônia (RO)
        state_sequences['RO'] = [
            ('NBB', 'NEH'), ('OHL', 'OHW'), ('OXL', 'OXL'),
            ('QRA', 'QRA'), ('QTA', 'QTJ'), ('RSU', 'RSZ'),
            ('SLG', 'SLL'), ('THI', 'THM'), ('UAJ', 'UAR')
        ]
        
        # Amapá (AP)
        state_sequences['AP'] = [
            ('NEI', 'NFB'), ('QLN', 'QLT'), ('SAK', 'SAM'),
            ('TGO', 'TGQ'), ('UKO', 'UKQ')
        ]
        
        return state_sequences
    
    def _get_state_codes(self):
        """Helper function to extract state codes from the mapping."""
        return self.state_digit_mapping
    
    def is_sequence_in_range(self, sequence: str, start: str, end: str) -> bool:
        """Checks if a sequence is within a range (alphabetical/numerical)."""
        # Handling the special format like SAV0A01.
        if 'SAV' in start and 'SAV' in end:
            return start <= sequence <= end
        
        # For regular 3-letter sequences.
        return start <= sequence <= end
    
    def get_state_from_sequence(self, sequence: str) -> Optional[str]:
        """Determines which state a license plate sequence belongs to."""
        sequence = sequence.upper()
        
        # Checking the Mercosur format (AAA9A99).
        if len(sequence) >= 3:
            first_three = sequence[:3]
            
            for state, ranges in self.state_sequences.items():
                for start, end in ranges:
                    if self.is_sequence_in_range(first_three, start[:3], end[:3]):
                        return state
        
        return None
    
    def validate_plate_format(self, plate_text: str) -> Tuple[bool, str]:
        """Validating if the plate follows Brazilian formats."""
        plate_text = plate_text.upper().strip()
        
        # Checking Mercosur format first (newer).
        if re.match(self.brazilian_patterns['mercosur']['format'], plate_text):
            return True, 'mercosur'
        
        # Checking the old format.
        if re.match(self.brazilian_patterns['old_system']['format'], plate_text):
            return True, 'old_system'
        
        return False, 'invalid'
    
    def is_valid_character(self, char: str, position: int, format_type: str) -> bool:
        """Checks if character is valid for its position in the plate."""
        if format_type == 'mercosur':
            if position in [0, 1, 2, 4]:  # Letter positions in Mercosur.
                return char in self.brazilian_patterns['mercosur']['valid_letters']
            elif position in [3, 5, 6]:  # Digit positions in Mercosur.
                return char in self.brazilian_patterns['mercosur']['valid_digits']
        else:  # old_system
            if position in [0, 1, 2]:  # First three are letters.
                return char in self.brazilian_patterns['old_system']['valid_letters']
            elif position in [3, 4, 5, 6]:  # Last four are digits.
                return char in self.brazilian_patterns['old_system']['valid_digits']
        
        return False
    
    def get_confusion_set(self, char: str) -> List[str]:
        """Gets the commonly confused characters for a given character."""
        char = char.upper()
        return self.confusion_matrix.get(char, [])
    
    def get_suggested_corrections(self, char: str) -> List[str]:
        """Gets suggested corrections for a character based on confusion matrix"""
        char = char.upper()
        suggestions = [char]  # Keep original as first option.
        if char in self.confusion_matrix:
            suggestions.extend(self.confusion_matrix[char])
        return suggestions

class PlateOCRCorrector:
    """OCR corrector tailored for Brazilian plates."""
    
    def __init__(self):
        self.patterns = BrazilianPlatePatterns()
        
    def parse_prediction_file(self, file_path: str) -> Dict:
        """Parses a single OCR prediction file."""
        predictions = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extracts the prediction.
            pred_match = re.search(r'PREDICTION:\s*(\w+)', content)
            if pred_match:
                predictions['prediction'] = pred_match.group(1).upper()
            
            # Extracts the confidence.
            conf_match = re.search(r'AVERAGE CONFIDENCE:\s*([\d.]+)%', content)
            if conf_match:
                predictions['avg_confidence'] = float(conf_match.group(1))
            
            # Extracts character-level analysis.
            char_pattern = r"Position \d+: '([A-Z0-9])' \(Confidence: ([\d.]+)%\)"
            char_matches = re.findall(char_pattern, content)
            
            predictions['characters'] = []
            predictions['confidences'] = []
            
            for char, conf in char_matches:
                predictions['characters'].append(char)
                predictions['confidences'].append(float(conf))
            
            # Extracts alternative OCR options.
            alt_pattern = r"'([A-Z0-9])': ([\d.]+)%"
            alt_sections = re.findall(r'Alternatives:(.*?)(?=\n\n|\nPosition|\Z)', content, re.DOTALL)
            
            predictions['alternatives'] = []
            for section in alt_sections:
                alts = re.findall(alt_pattern, section)
                alt_dict = {char: float(conf) for char, conf in alts}
                predictions['alternatives'].append(alt_dict)
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return predictions
    
    def weighted_voting_with_state_validation(self, all_predictions: List[Dict]) -> str:
        """Performs weighted voting with state sequence validation"""
        if not all_predictions:
            return ""
        
        # Initializing weights based on average confidence.
        weights = []
        normalized_predictions = []
        
        for pred in all_predictions:
            if 'prediction' in pred and 'avg_confidence' in pred:
                weight = pred['avg_confidence'] / 100.0
                plate_text = pred['prediction']
                
                # Normalizing length - pad or truncating to 7 characters.
                if len(plate_text) < 7:
                    plate_text = plate_text.ljust(7, '0')
                elif len(plate_text) > 7:
                    plate_text = plate_text[:7]
                
                weights.append(weight)
                normalized_predictions.append(plate_text)
        
        if not normalized_predictions:
            return ""
        
        # Weighted voting per position.
        final_plate = []
        position_weights = []
        
        for pos in range(7):
            pos_chars = []
            pos_weights = []
            pos_confidences = []
            
            for idx, plate in enumerate(normalized_predictions):
                if pos < len(plate):
                    char = plate[pos]
                    weight = weights[idx]
                    
                    # Gets confidence for this specific character if available.
                    char_conf = 1.0
                    if 'confidences' in all_predictions[idx] and pos < len(all_predictions[idx]['confidences']):
                        char_conf = all_predictions[idx]['confidences'][pos] / 100.0
                    
                    pos_chars.append(char)
                    pos_weights.append(weight * char_conf)
                    pos_confidences.append(char_conf)
            
            if pos_chars:
                # Weighted voting with consideration of confusion sets.
                char_counter = defaultdict(float)
                char_confidence_sum = defaultdict(float)
                char_count = defaultdict(int)
                
                for char, weight, conf in zip(pos_chars, pos_weights, pos_confidences):
                    char_counter[char] += weight
                    char_confidence_sum[char] += conf
                    char_count[char] += 1
                
                # Also considering confusion characters.
                for char in list(char_counter.keys()):
                    confusion_set = self.patterns.get_confusion_set(char)
                    for confused_char in confusion_set:
                        # Distributing some weight to commonly confused characters.
                        char_counter[confused_char] += char_counter[char] * 0.2
                
                # Selects character with highest weighted score
                best_char = max(char_counter.items(), key=lambda x: x[1])[0]
                final_plate.append(best_char)
                
                # Calculates position confidence.
                if char_counter:
                    total_weight = sum(char_counter.values())
                    if total_weight > 0:
                        pos_conf = char_counter[best_char] / total_weight * 100
                        position_weights.append(pos_conf)
            else:
                final_plate.append('0')
                position_weights.append(0.0)
        
        result = ''.join(final_plate)
        
        # Validating the state sequence and adjusting if necessary.
        state = self.patterns.get_state_from_sequence(result[:3])
        if state is None:
            # Trying to correct first three letters based on state sequences.
            result = self.correct_state_sequence(result)
        
        return result
    
    def correct_state_sequence(self, plate_text: str) -> str:
        """Corrects the state sequence (first 3 letters) based on known sequences."""
        if len(plate_text) < 3:
            return plate_text
        
        first_three = plate_text[:3]
        rest = plate_text[3:]
        
        # Checks if current sequence is valid.
        state = self.patterns.get_state_from_sequence(first_three)
        if state is not None:
            return plate_text
        
        # Attempting to find the closest valid sequence.
        best_correction = first_three
        best_score = 0
        
        for state_code, sequences in self.patterns.state_sequences.items():
            for start, end in sequences:
                # Check if we should consider this range.
                if len(start) == 3 and len(end) == 3:
                    # Generate some test sequences around the input.
                    test_sequences = []
                    
                    # Trying the original.
                    test_sequences.append(first_three)
                    
                    # Trying confusion-based corrections.
                    for i in range(3):
                        suggestions = self.patterns.get_suggested_corrections(first_three[i])
                        for suggestion in suggestions[:2]:  # Top 2 suggestions.
                            test_seq = list(first_three)
                            test_seq[i] = suggestion
                            test_sequences.append(''.join(test_seq))
                    
                    # Checking each test sequence.
                    for test_seq in test_sequences:
                        if start <= test_seq <= end:
                            # Calculating the similarity score.
                            score = sum(1 for a, b in zip(first_three, test_seq) if a == b)
                            if score > best_score:
                                best_score = score
                                best_correction = test_seq
        
        return best_correction + rest
    
    def apply_format_rules(self, plate_text: str) -> str:
        """Applies Brazilian plate format rules to correct plate."""
        plate_text = plate_text.upper().strip()
        
        # Checks which format it might be.
        is_valid, format_type = self.patterns.validate_plate_format(plate_text)
        
        if is_valid:
            return plate_text
        
        # Trying to correct based on format.
        corrected = list(plate_text)
        
        # Mercosur format correction.
        if len(corrected) >= 7:
            # Ensuring proper character types per position.
            for i in range(7):
                if i in [0, 1, 2, 4]:  # Letters.
                    if not corrected[i].isalpha():
                        # Trying to find the most likely letter from the confusion set.
                        confusion = self.patterns.get_confusion_set(corrected[i])
                        if confusion:
                            corrected[i] = confusion[0]
                        else:
                            corrected[i] = 'A'  # Default.
                elif i in [3, 5, 6]:  # Digits.
                    if not corrected[i].isdigit():
                        confusion = self.patterns.get_confusion_set(corrected[i])
                        if confusion and any(c.isdigit() for c in confusion):
                            digit_confs = [c for c in confusion if c.isdigit()]
                            corrected[i] = digit_confs[0]
                        else:
                            corrected[i] = '0'  # Default.
        
        return ''.join(corrected)
    
    def consensus_correction(self, all_predictions: List[Dict]) -> str:
        """Advanced consensus-based correction using all 5 predictions"""
        # Weighted voting result with state validation.
        weighted_result = self.weighted_voting_with_state_validation(all_predictions)
        
        # Simple majority vote
        predictions_list = [p.get('prediction', '').upper() for p in all_predictions if p.get('prediction')]
        if predictions_list:
            majority_result = self.majority_vote(predictions_list)
        else:
            majority_result = weighted_result
        
        # Applying format rules to both.
        weighted_corrected = self.apply_format_rules(weighted_result)
        majority_corrected = self.apply_format_rules(majority_result)
        
        # Choosing the one that better matches Brazilian formats.
        weighted_valid, weighted_format = self.patterns.validate_plate_format(weighted_corrected)
        majority_valid, majority_format = self.patterns.validate_plate_format(majority_corrected)
        
        # Checking the state sequence validity.
        weighted_state = self.patterns.get_state_from_sequence(weighted_corrected[:3])
        majority_state = self.patterns.get_state_from_sequence(majority_corrected[:3])
        
        if weighted_valid and not majority_valid:
            return weighted_corrected
        elif majority_valid and not weighted_valid:
            return majority_corrected
        elif weighted_valid and majority_valid:
            # Both valid, consider state sequence.
            if weighted_state and not majority_state:
                return weighted_corrected
            elif majority_state and not weighted_state:
                return majority_corrected
            elif weighted_state and majority_state:
                # Both have valid state sequences, choose based on confidence.
                weighted_conf = np.mean([p.get('avg_confidence', 0) for p in all_predictions if 'avg_confidence' in p])
                return weighted_corrected if weighted_conf >= 70 else majority_corrected
            else:
                # Neither has valid state sequence, choose based on format.
                if weighted_format == 'mercosur' and majority_format != 'mercosur':
                    return weighted_corrected
                else:
                    return majority_corrected
        else:
            # Neither valid, return the weighted one with format applied.
            return weighted_corrected
    
    def majority_vote(self, predictions: List[str]) -> str:
        """Simple majority voting."""
        if not predictions:
            return ""
        
        # Pad all to same length.
        max_len = max(len(p) for p in predictions)
        padded = [p.ljust(max_len, '0') for p in predictions]
        
        # Vote per position.
        result = []
        for pos in range(max_len):
            chars = [p[pos] for p in padded]
            char_counts = Counter(chars)
            most_common = char_counts.most_common(1)[0][0]
            result.append(most_common)
        
        return ''.join(result)[:7]  # Ensures max 7 characters.
    
    def calculate_accuracy(self, ground_truth: str, prediction: str, exact_match: bool = True) -> float:
        """Calculates accuracy between ground truth and prediction.
        
        Args:
            ground_truth: The true plate text
            prediction: The predicted plate text
            exact_match: If True, requires all characters to match (returns 0% or 100%)
                        If False, returns character-level accuracy
        """
        gt = ground_truth.upper().strip()
        pred = prediction.upper().strip()
        
        if exact_match:
            # EXACT MATCH: All characters must be correct.
            return 100.0 if gt == pred else 0.0
        else:
            # CHARACTER-LEVEL ACCURACY: Percentage of correct characters.
            # Handling different lengths.
            max_len = max(len(gt), len(pred))
            gt_padded = gt.ljust(max_len, ' ')
            pred_padded = pred.ljust(max_len, ' ')
            
            # Character-level accuracy.
            correct_chars = sum(1 for g, p in zip(gt_padded, pred_padded) if g == p)
            accuracy = correct_chars / max_len if max_len > 0 else 0
            
            return accuracy * 100
    
    def process_plate_set(self, prediction_files: List[str], ground_truth: str) -> Dict:
        """Process a set of 5 prediction files for one plate."""
        all_predictions = []
        
        # Parsing all prediction files.
        for file_path in prediction_files:
            pred_data = self.parse_prediction_file(file_path)
            if pred_data:
                all_predictions.append(pred_data)
        
        # Getting individual predictions.
        individual_preds = [p.get('prediction', '') for p in all_predictions]
        
        # Applying consensus correction.
        final_prediction = self.consensus_correction(all_predictions)
        
        # Determining the state.
        state = self.patterns.get_state_from_sequence(final_prediction[:3])
        
        # Accuracy calculation.
        accuracy = self.calculate_accuracy(ground_truth, final_prediction)
        
        # Individual accuracy calculation.
        individual_accuracies = [
            self.calculate_accuracy(ground_truth, pred) 
            for pred in individual_preds
        ]
        
        # Determining the plate format.
        is_valid, format_type = self.patterns.validate_plate_format(final_prediction)
        
        return {
            'ground_truth': ground_truth,
            'final_prediction': final_prediction,
            'accuracy': accuracy,
            'individual_predictions': individual_preds,
            'individual_accuracies': individual_accuracies,
            'avg_individual_accuracy': np.mean(individual_accuracies) if individual_accuracies else 0,
            'plate_format': format_type if is_valid else 'invalid',
            'is_valid_format': is_valid,
            'state': state,
            'confidence_scores': [p.get('avg_confidence', 0) for p in all_predictions],
            'num_predictions': len(prediction_files)
        }


def batch_process_all_plates(predictions_dir, ground_truth_dir, output_file="all_results.csv"):
    """Batch processes all 500 plates."""
    from tqdm import tqdm
    
    corrector = PlateOCRCorrector()
    
    all_results = []
    failed_plates = []
    
    print("=" * 80)
    print("BATCH PROCESSING BRAZILIAN LICENSE PLATES")
    print("=" * 80)
    
    for plate_num in tqdm(range(500), desc="Processing plates"):
        gt_file = os.path.join(ground_truth_dir, f"{plate_num}_gt.txt")
        
        if not os.path.exists(gt_file):
            failed_plates.append((plate_num, "Ground truth file not found"))
            continue
        
        # Reading the ground truth.
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                ground_truth = f.read().strip()
        except Exception as e:
            failed_plates.append((plate_num, f"Error reading ground truth: {e}"))
            continue
        
        # Looking for the prediction files for each scenario.
        prediction_files = []
        for suffix in ['lr-001', 'lr-002', 'lr-003', 'lr-004', 'lr-005']:
            pred_file = os.path.join(predictions_dir, f"{plate_num}_{suffix}_report.txt")
            if os.path.exists(pred_file):
                prediction_files.append(pred_file)
        
        if prediction_files:
            try:
                result = corrector.process_plate_set(prediction_files, ground_truth)
                result['plate_id'] = plate_num
                
                result['exact_match_accuracy'] = corrector.calculate_accuracy(
                    ground_truth, result['final_prediction'], exact_match=True
                )
                result['character_accuracy'] = corrector.calculate_accuracy(
                    ground_truth, result['final_prediction'], exact_match=False
                )
                
                individual_exact_accuracies = [
                    corrector.calculate_accuracy(ground_truth, pred, exact_match=True)
                    for pred in result['individual_predictions']
                ]
                result['individual_exact_accuracies'] = individual_exact_accuracies
                result['avg_individual_exact_accuracy'] = np.mean(individual_exact_accuracies) if individual_exact_accuracies else 0
                
                all_results.append(result)
            except Exception as e:
                failed_plates.append((plate_num, f"Processing error: {e}"))
        else:
            failed_plates.append((plate_num, "No prediction files found"))
    
    # Converting to DataFrame and saving.
    if all_results:
        df = pd.DataFrame(all_results)
        
        exact_match_success = df['exact_match_accuracy'] == 100
        exact_match_rate = exact_match_success.mean() * 100
        
        avg_character_accuracy = df['character_accuracy'].mean()
        
        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total plates processed successfully: {len(df)}")
        print(f"Plates failed to process: {len(failed_plates)}")
        
        print(f"\n=== EXACT MATCH ACCURACY (All characters correct) ===")
        print(f"  Exact match rate: {exact_match_rate:.2f}%")
        print(f"  Correct plates: {exact_match_success.sum()}/{len(df)}")
        
        print(f"\n=== CHARACTER-LEVEL ACCURACY ===")
        print(f"  Average character accuracy: {avg_character_accuracy:.2f}%")
        
        print(f"\n=== IMPROVEMENT ANALYSIS ===")

        individual_exact_rates = []
        for idx, row in df.iterrows():
            if row['individual_exact_accuracies']:
                best_individual = max(row['individual_exact_accuracies'])
                individual_exact_rates.append(best_individual)
        
        avg_best_individual_exact = np.mean(individual_exact_rates) if individual_exact_rates else 0
        exact_improvement = exact_match_rate - avg_best_individual_exact
        
        print(f"  Best individual exact match rate: {avg_best_individual_exact:.2f}%")
        print(f"  Exact match improvement: {exact_improvement:.2f} percentage points")
        
        df.to_csv(output_file, index=False)
        
        summary_report = {
            'total_plates': 500,
            'successfully_processed': len(df),
            'failed_plates': len(failed_plates),
            'exact_match_rate': float(exact_match_rate),
            'character_level_accuracy': float(avg_character_accuracy),
            'improvement_stats': {
                'best_individual_exact_rate': float(avg_best_individual_exact),
                'exact_match_improvement': float(exact_improvement),
                'plates_correct_exact': int(exact_match_success.sum()),
                'plates_improved_exact': int((df['exact_match_accuracy'] > df['avg_individual_exact_accuracy']).sum()),
                'plates_worse_exact': int((df['exact_match_accuracy'] < df['avg_individual_exact_accuracy']).sum())
            }
        }
        
        with open("processing_summary_exact_match.json", 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        print(f"Summary report saved to processing_summary_exact_match.json")
        
        return df
    else:
        print("No plates were successfully processed.")
        return None

def main():
    predictions_dir = "C:/Users/bgat/Desktop/Antonis/CALAMARI_PIPELINE_OUTPUT/reports"
    ground_truth_dir = "C:/Users/bgat/Desktop/Antonis"
    output_file = "brazilian_plate_results.csv"
    
    print("Starting batch processing of Brazilian license plates...")
    results = batch_process_all_plates(predictions_dir, ground_truth_dir, output_file)
    
    if results is not None:
        print("\n" + "=" * 80)
        print("SAMPLE RESULTS (First 10 plates):")
        print("=" * 80)
        
        sample = results.head(10)
        for _, row in sample.iterrows():
            print(f"\nPlate {row['plate_id']}:")
            print(f"  Ground Truth: {row['ground_truth']}")
            print(f"  Final Prediction: {row['final_prediction']}")
            print(f"  Accuracy: {row['accuracy']:.2f}%")
            print(f"  Format: {row['plate_format']}")
            print(f"  State: {row['state']}")
        
        # Shows the top 5 most improved plates.
        if 'accuracy' in results.columns and 'avg_individual_accuracy' in results.columns:
            results['improvement'] = results['accuracy'] - results['avg_individual_accuracy']
            top_improved = results.nlargest(5, 'improvement')
            
            print("\n" + "=" * 80)
            print("TOP 5 MOST IMPROVED PLATES:")
            print("=" * 80)
            
            for _, row in top_improved.iterrows():
                print(f"\nPlate {row['plate_id']}:")
                print(f"  Ground Truth: {row['ground_truth']}")
                print(f"  Final Prediction: {row['final_prediction']}")
                print(f"  Individual Accuracy: {row['avg_individual_accuracy']:.2f}%")
                print(f"  Final Accuracy: {row['accuracy']:.2f}%")
                print(f"  Improvement: {row['improvement']:.2f}%")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
