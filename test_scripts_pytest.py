import os

import pytest
from Bio.Seq import Seq

from bio_files_processor import (convert_multiline_fasta_to_oneline,
                                 FastaRecord, OpenFasta)
from bi_python_kit import (calculate_gc_content, DNASequence,
                         RNASequence, AminoAcidSequence,
                         run_genscan, GenscanOutput)


class TestSequenceFunctions:
    def test_calculate_gc_content(self):
        seq = Seq("ATGCATGC")
        assert calculate_gc_content(seq) == 50.0

    def test_transcribe_method(self):
        dna_sequence = DNASequence("ATGC")
        rna_sequence = dna_sequence.transcribe()
        assert isinstance(rna_sequence, RNASequence)
        assert str(rna_sequence) == "AUGC"

    def test_complement_sequence(self):
        dna_sequence = DNASequence("ATGC")
        complement_sequence = dna_sequence.complement()
        assert str(complement_sequence) == "TACG"

    def test_calculate_molecular_weight(self):
        aa_sequence = AminoAcidSequence("ARKEN")
        assert aa_sequence.calculate_molecular_weight() == 598.653


class TestFastaFunctions:
    @pytest.fixture
    def input_fasta(self, tmpdir, request):
        if request.node.name == "test_convert_multiline_fasta_to_oneline":
            fasta_content = [
                ">seq1 Descr1",
                "ATCG",
                "CCTG",
                ">seq2 Descr2",
                "GCTA",
                "CGAT"
            ]
        elif request.node.name == "test_open_fasta_reading":
            fasta_content = [
                ">seq1 Descr1",
                "ATCG",
                ">seq2 Descr2",
                "GCTA"
            ]
        fasta_file = tmpdir.join("input.fasta")
        fasta_file.write("\n".join(fasta_content))
        return str(fasta_file)

    @pytest.fixture
    def output_fasta(self, tmpdir):
        return str(tmpdir.join("output.fasta"))

    def test_convert_multiline_fasta_to_oneline(self, input_fasta, output_fasta):
        convert_multiline_fasta_to_oneline(input_fasta, output_fasta)
        assert os.path.exists(output_fasta)
        expected_output = [
            ">seq1 Descr1\n",
            "ATCGCCTG\n",
            ">seq2 Descr2\n",
            "GCTACGAT\n"
        ]
        with open(output_fasta, "r") as f:
            lines = f.readlines()
            assert lines == expected_output

    def test_open_fasta_reading(self, input_fasta):
        with OpenFasta(input_fasta) as fasta_reader:
            records = fasta_reader.read_records()
        assert len(records) == 2
        assert records[0] == FastaRecord(seq_id=">seq1", description="Descr1", seq="ATCG")
        assert records[1] == FastaRecord(seq_id=">seq2", description="Descr2", seq="GCTA")


class TestGenscanFunctions:
    @pytest.fixture
    def sequence_file_path(self):
        return "./data/example_genscan.fa"

    def test_run_genscan_with_sequence_file(self, sequence_file_path):
        result = run_genscan(sequence_file=sequence_file_path)
        assert isinstance(result, GenscanOutput)
        assert result.status == 200
        assert len(result.cds_list) == 1
        assert len(result.exon_dict) == 6
        assert len(result.intron_dict) == 5

    def test_run_genscan_without_sequence(self):
        with pytest.raises(ValueError):
            run_genscan(sequence="", sequence_file="")
