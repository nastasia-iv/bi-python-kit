import datetime
import io
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Type

import requests
from Bio import SeqIO, Seq, SeqUtils
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


class SequenceError(ValueError):
    """
    Exception raised for invalid nucleic acid sequences.
    """

    def __init__(self, message="Operation cannot be performed: incorrect sequence."):
        self.message = message
        super().__init__(self.message)


class BiologicalSequence(ABC):
    """
    Abstract class for a biological sequence.
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def alphabet_is_valid(self) -> bool:
        """
        Check if the sequence's alphabet is valid.

        Arguments:
            self: the sequence object to check

        Returns:
            bool: True if the sequence contains valid characters, False otherwise.
        """
        pass


class NucleicAcidSequence(BiologicalSequence):
    """
    Class for a nucleic acid sequence.
    Created only to set methods of DNASequence and RNASequence classes.
    """

    def __init__(self, sequence: str):
        self.sequence = sequence

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, index) -> str:
        return self.sequence[index]

    def __str__(self) -> str:
        return self.sequence

    def alphabet_is_valid(self) -> bool:
        """
        Check if the sequence's alphabet is valid.

        Returns:
            bool: True if the sequence contains valid characters, False otherwise.
        """
        return all(nucleotide in self.alphabet for nucleotide in self.sequence)

    def complement(self) -> Type['BiologicalSequence']:
        """
        Generate the complementary sequence.

        Returns:
            Type['BiologicalSequence']: Complementary sequence.
        """
        if not self.alphabet_is_valid():
            raise SequenceError()

        complement_pairs_dna = {
            'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
            'a': 't', 't': 'a', 'g': 'c', 'c': 'g'
        }
        complement_pairs_rna = {
            'G': 'C', 'C': 'G', 'U': 'A', 'A': 'U',
            'g': 'c', 'c': 'g', 'u': 'a', 'a': 'u'
        }

        complement_sequence = ''
        complement_pairs = complement_pairs_dna if isinstance(self, DNASequence) else complement_pairs_rna

        for base in self.sequence:
            complement_sequence += complement_pairs.get(base, base)
        return type(self)(complement_sequence)

    def gc_content(self) -> float:
        """
        Calculate the GC content of the sequence.

        Returns:
            float: GC content as a percent.
        """
        if not self.alphabet_is_valid():
            raise SequenceError()
        return (sum(nucleotide in "CcGg" for nucleotide in self.sequence) / len(self)) * 100


class DNASequence(NucleicAcidSequence):
    """
    Class for a DNA sequence.
    """
    alphabet = "ATGCatgc"

    def __init__(self, sequence: str):
        self.sequence = sequence

    def transcribe(self) -> 'RNASequence':
        """
        Transcribe the DNA sequence into an RNA sequence.

        Returns:
            RNASequence: Transcribed RNA sequence.
        """
        if self.alphabet_is_valid():

            transcribe_sequence = ''
            for base in self.sequence:
                if base == 'T':
                    transcribe_sequence += 'U'
                elif base == 't':
                    transcribe_sequence += 'u'
                else:
                    transcribe_sequence += base
            return RNASequence(transcribe_sequence)

        raise SequenceError()


class RNASequence(NucleicAcidSequence):
    """
    Class for an RNA sequence.
    """
    alphabet = "AUGCaugc"

    def __init__(self, sequence: str):
        self.sequence = sequence


class AminoAcidSequence(BiologicalSequence):
    """
    Class for an amino acid sequence.
    """
    amino_acid_weights = {
        'G': 57.051, 'A': 71.078, 'S': 87.077, 'P': 97.115, 'V': 99.131,
        'T': 101.104, 'C': 103.143, 'I': 113.158, 'L': 113.158, 'N': 114.103,
        'D': 115.087, 'Q': 128.129, 'K': 128.172, 'E': 129.114, 'M': 131.196,
        'H': 137.139, 'F': 147.174, 'R': 156.186, 'Y': 163.173, 'W': 186.210
    }

    def __init__(self, sequence: str):
        self.sequence = sequence

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, index) -> str:
        return self.sequence[index]

    def __str__(self) -> str:
        return self.sequence

    def alphabet_is_valid(self) -> bool:
        """
        Check if the amino acid sequence's alphabet is valid.

        Returns:
            bool: True if the sequence contains valid characters, False otherwise.
        """
        amino_acid_alphabet = set("ARNDCHGQEILKMPSYTWFV")
        return set(self.sequence).issubset(amino_acid_alphabet)

    def calculate_molecular_weight(self) -> float:
        """
        Calculate the molecular weight of the amino acid sequence.

        Returns:
            float: Molecular weight of the sequence.
        """
        return sum(self.amino_acid_weights[aa] for aa in self.sequence)


def calculate_gc_content(seq: Seq) -> float:
    """
    Calculate the GC content of a nucleotide sequence.

    Arguments:
        seq (Seq): Biological sequence.

    Returns:
        float: GC content as a percentage.
    """
    return SeqUtils.GC123(seq)[0]


def calculate_average_quality(seq: Seq) -> float:
    """
    Calculate the average quality of a nucleotide sequence.

    Arguments:
        seq (Seq): Biological sequence.

    Returns:
        float: Average sequence quality according to the phred33 scale.
    """
    return sum(seq.letter_annotations["phred_quality"]) / len(seq)  # letter_annotations from Bio.SeqRecord


def save_records_to_fastq(filtered_records: List[SeqIO.SeqRecord],
                          output_filename: str, input_path: str) -> str:
    """
    Save filtered records to a new FASTQ file.

    Arguments:
        filtered_records (List[SeqIO.SeqRecord]): List of filtered SeqRecords.
        output_filename (str): Name of the file to save.
        input_path (str): Path to the original FASTQ file.

    Returns:
        str: Path to the saved file.
    """
    if output_filename == '':
        output_filename = os.path.basename(input_path)
    if not output_filename.endswith(".fastq"):
        output_filename += ".fastq"
    output_dir = 'fastq_filtrator_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    SeqIO.write(filtered_records, output_path, "fastq")

    return output_path


def filter_fastq(input_path: str,
                 output_filename: str = '',
                 gc_bounds: Union[tuple, int, float] = (0, 100),
                 length_bounds: Union[tuple, int] = (0, 2 ** 32),
                 quality_threshold: int = 0) -> str:
    """
    Filter FASTQ records based on specified criteria and save them to a new file.

    Arguments:
        input_path (str): Path to the FASTQ file.
        output_filename (str): Optional, file name to save the result.
        gc_bounds (Union[tuple, int, float]): GC composition interval (percents) to filter, default is (0, 100).
        length_bounds (Union[tuple, int]): Sequence length interval to filter, default is (0, 2**32).
        quality_threshold (int): Threshold value of average read quality (phred33) to filter, default is 0.

    Returns:
        str: Path for filtered file.
    """
    records = list(SeqIO.parse(input_path, "fastq"))

    filtered_records = []

    if isinstance(gc_bounds, (float, int)):
        gc_min = 0
        gc_max = gc_bounds
    else:
        gc_min = gc_bounds[0]
        gc_max = gc_bounds[1]

    if isinstance(length_bounds, int):
        length_min = 0
        length_max = length_bounds
    else:
        length_min = length_bounds[0]
        length_max = length_bounds[1]

    for record in records:
        gc_content = calculate_gc_content(record.seq)
        length = len(record)
        avg_quality = calculate_average_quality(record)

        if (
                gc_min <= gc_content <= gc_max
                and length_min <= length <= length_max
                and avg_quality >= quality_threshold
        ):
            filtered_records.append(record)

    return save_records_to_fastq(filtered_records, output_filename, input_path)


def format_time(timedelta: datetime.timedelta) -> str:
    """
    Format a timedelta object into a human-readable string.

    Arguments:
        timedelta (datetime.timedelta): The timedelta object to be formatted.

    Returns:
        str: The formatted string representing the timedelta.
    """
    if timedelta.days == 0:
        return str(timedelta)
    return timedelta.strftime("%-d days, %H:%M:%S")


def write_log_and_send_telegram_message(chat_id: str, message: str, log_content: str, log_filename: str) -> None:
    """
    Write log to a file, then send it as a document in a Telegram bot.

    Arguments:
        chat_id (str): The ID of the Telegram bot to send the message to.
        message (str): The text of the message.
        log_content (str): The content to be written to the log file.
        log_filename (str): The filename to use for the log file.
    """

    with open(log_filename, "w") as log_file:
        log_file.write(log_content)
    with open(log_filename, "rb") as log_file:
        url = f"https://api.telegram.org/bot{os.getenv('TG_API_TOKEN')}/sendDocument"
        params = {"chat_id": chat_id, "caption": message, "parse_mode": "Markdown"}
        if log_file:
            files = {"document": log_file}
            response = requests.post(url, params=params, files=files)
        else:
            response = requests.post(url, params=params)
        if response.status_code != 200:
            print(f"Error with sending message to Telegram: {response.text}")

    os.remove(log_filename)


def telegram_logger(chat_id: str):
    """
    Decorator for logging function execution and sending logs to Telegram bot.

    Arguments:
        chat_id (str): The ID of the Telegram chat to send the logs to.

    Returns:
        function: Decorator function.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            stdout_backup = sys.stdout
            stderr_backup = sys.stderr

            # Create empty buffers in memory
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Capture output and errors instead of standard streams
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                error_message = f"😞 Function `{func.__name__}` failed with an exception:\n`{type(e).__name__}: {e}`"
                stdout_content = stdout_capture.getvalue()
                stderr_content = stderr_capture.getvalue()
                combined_output = stdout_content + stderr_content
                log_filename = f"{func.__name__}.log"
                write_log_and_send_telegram_message(chat_id, error_message, combined_output, log_filename)
                raise

            finally:
                sys.stdout = stdout_backup
                sys.stderr = stderr_backup

            end_time = datetime.datetime.now()
            execution_time = format_time(end_time - start_time)
            success_message = f"🎉 Function `{func.__name__}` successfully finished in `{execution_time}`"
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            combined_output = stdout_content + stderr_content
            log_filename = f"{func.__name__}.log"
            write_log_and_send_telegram_message(chat_id, success_message, combined_output, log_filename)

            sys.stdout = stdout_backup
            sys.stderr = stderr_backup

            return result
        return wrapper
    return decorator


@dataclass
class GenscanOutput:
    """
    Represents the output of the GENSCAN prediction.

    Attributes:
        status (str): The status of the prediction.
        cds_list (List[str]): List of predicted protein sequences taking splicing into account.
        exon_dict (Dict[str, Tuple[str, str]]): Dictionary of exons with their start and end positions.
        intron_dict (Dict[str, Tuple[str, str]]): Dictionary of introns with their start and end positions.
    """
    status: int
    cds_list: List[str]
    exon_dict: Dict[str, Tuple[int, int]]
    intron_dict: Dict[str, Tuple[int, int]]


def run_genscan(sequence: Optional[str] = None, sequence_file: Optional[str] = None,
                organism: str = "Vertebrate", exon_cutoff: float = 1.00,
                sequence_name: str = "") -> GenscanOutput:
    """
    Run Genscan prediction based on the provided DNA sequence or file path.

    Arguments:
        sequence (str, optional): The DNA sequence. Defaults to None.
        sequence_file (str, optional): The path to the file containing the DNA sequence. Defaults to None.
        organism (str, optional): The organism type for prediction. Defaults to "Vertebrate".
        exon_cutoff (float, optional): The cutoff value for exons. Defaults to 1.00.
        sequence_name (str, optional): The name of the sequence. Defaults to "".

    Returns:
        GenscanOutput: An instance of GenscanOutput containing the prediction results.
    """

    if not sequence and not sequence_file:
        raise ValueError("Please, specify the sequence or the path to the file with the sequence!")

    if sequence_file:
        with open(sequence_file, 'r') as file:
            sequence = file.read()

    url = 'http://hollywood.mit.edu/cgi-bin/genscanw_py.cgi'
    data = {
        '-o': organism,
        '-e': exon_cutoff,
        '-n': sequence_name,
        '-p': 'Predicted peptides only',
        '-u': '(binary)',
        '-s': sequence
    }
    response = requests.post(url,
                             data=data)

    if response.status_code != 200:
        raise ConnectionError("Error with sending request!")

    status = response.status_code

    soup = BeautifulSoup(response.text, 'html.parser')
    info_str = soup.find('pre').string

    # Get a list of predicted protein sequences taking into account splicing
    cds_raw_data = info_str.split('Predicted peptide sequence(s):')[-1].split('>')
    cds_list = []

    for peptide in cds_raw_data[1:]:
        cds_lines = peptide.split('\n')
        cds_sequence = ''
        for line in cds_lines:
            if not line.startswith('/tmp'):
                cds_sequence += line.strip()
        cds_list.append(cds_sequence)

    # Get a list of exons and introns
    exon_intron_raw_data = info_str.split('Suboptimal exons with probability')
    table = exon_intron_raw_data[0].split('\n\n\n\n')

    records_list = []
    for element in table[5:]:
        record = element.strip().split('\n\n')
        records_list.extend(record)

    records_list = records_list[:-1]

    exons = []
    introns = []
    intron_dict = {}

    for i in range(len(records_list) - 1):
        exon = records_list[i]
        next_exon = records_list[i + 1]

        processed_exon = re.sub(r'\s+', ' ', exon.strip()).split(' ')
        processed_next_exon = re.sub(r'\s+', ' ', next_exon.strip()).split(' ')

        exons.append(processed_exon)

        end_exon = int(processed_exon[4])
        start_next_exon = int(processed_next_exon[3])

        start_intron = end_exon + 1
        end_intron = start_next_exon - 1

        introns.append((start_intron, end_intron))
        intron_dict[f'Intron {processed_exon[0]}'] = (start_intron, end_intron)

    # Add information about the last exon
    last_exon = records_list[-1]
    processed_last_exon = re.sub(r'\s+', ' ', last_exon.strip()).split(' ')
    exons.append(processed_last_exon)

    exon_dict = {f'Exon {exon[0]}': (int(exon[3]), int(exon[4])) for exon in exons}

    return GenscanOutput(status, cds_list, exon_dict, intron_dict)
