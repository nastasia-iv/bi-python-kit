from dataclasses import dataclass
from typing import Iterator, List, Tuple


def convert_multiline_fasta_to_oneline(input_fasta: str,
                                       output_fasta: str = '') -> None:
    """
    Converts multi-line FASTA sequences to single-line sequences.

    Reads a FASTA file in which some sequences are written on several lines,
    and writes it to a new file, where each sequence is written on one line.

    Arguments:
        input_fasta (str): Path to the FASTA file to read.
        output_fasta (str, optional): Name for the FASTA file. If not specified, a file with the input name and postfix '_oneline' is created.

    Returns:
        file with processed (single-line) FASTA sequences.
    """
    if output_fasta == '':
        output_fasta = input_fasta.replace('.fasta', '_oneline.fasta')
    else:
        if not output_fasta.endswith(".fasta"):
            output_fasta += ".fasta"
    output_lines = []  # final list
    with open(input_fasta, mode="r") as input_file:
        current_sequence = []  # temporary list for the lines of the current sequence
        for line in input_file:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence:
                    output_lines.append("".join(current_sequence))  # merge all sequences into one into the final list
                output_lines.append(line)  # add the identifier to the final list
                current_sequence = []  # clear the temporary list
            else:
                current_sequence.append(line)  # add line to the temporary list

        # Add sequences for the last identifier from the temporary list to the final one
        if current_sequence:
            output_lines.append("".join(current_sequence))
    with open(output_fasta, mode="w") as output_file:
        output_file.write("\n".join(output_lines))


def select_genes_from_gbk_to_fasta(input_gbk: str,
                                   genes: Tuple[str],
                                   n_before: int = 1,
                                   n_after: int = 1,
                                   output_fasta: str = '') -> None:
    """
    Selects neighbor genes for the gene of interest from the GBK file and writes their protein sequences into FASTA format.

     Arguments:
        input_gbk (str): path to the GBK file to read.
        genes (Tuple[str]): list of genes of interest.
        n_before (int, optional): number of genes before gene of interest. Defaults to 1.
        n_after (int, optional): number of genes after gene of interest. Defaults to 1.
        output_fasta (str, optional): path to the FASTA file to write to. If not specified, a file with the input name and postfix '_selected' is created.

    Returns:
        file with genes of interest from the GBK file.
     """
    if output_fasta == '':
        output_fasta = input_gbk.replace(".gbk", "_selected.fasta")
    else:
        if not output_fasta.endswith(".fasta"):
            output_fasta += ".fasta"

    # Block for creating a list of genes and a dictionary
    gene_protein_dict = {}
    with open(input_gbk, mode="r") as input_file:
        current_gene = None
        current_translation = None
        in_gene = False
        gene_list = []
        for line in input_file:
            line = line.strip()
            if line.startswith("/gene="):
                current_gene = line.split("=")[1].strip('"\n')
                line = line.replace('/gene="', '')
                line = line.strip('"')
                gene_list.append(line)
                in_gene = True
            elif in_gene and line.startswith("/translation="):  # if find a gene, look for the protein sequence
                current_translation = line.split("=")[1].strip('"')  # separator =, take only element 1, remove the quote at the end
                while not line.endswith('"'):  # until the end of the sequence is found
                    line = input_file.readline().strip()  # read the next line
                    current_translation += line.strip('"')
                in_gene = False  # reset the gene mark

                if current_gene and current_translation:  # if both the gene name and its protein sequence found, add them to the dictionary
                    gene_protein_dict[current_gene] = current_translation
                    current_gene = None
                    current_translation = None

    # Block for searching protein sequences of neighboring genes
    neighbor_gene_proteins = []  # to store a list of tuples (neighbor gene, sequence)
    for gene in genes:
        if any(gene in name for name in gene_protein_dict):
            index = gene_list.index(gene)  # find the index of the gene of interest in the general list
            # Calculate the indices of neighboring genes
            start = max(0, index - n_before)  # don't take negative values
            end = min(len(gene_list), index + n_after + 1)  # don't take values > gene list length
            for index in range(start, end):
                neighbor_gene = gene_list[index]
                if neighbor_gene != gene:
                    neighbor_gene_proteins.append((neighbor_gene, gene_protein_dict[neighbor_gene]))

    with open(output_fasta, mode="w") as output_file:
        for gene_and_translation in neighbor_gene_proteins:
            gene, translation = gene_and_translation  # unpack the list into tuples
            output_file.write(f">{gene}\n{translation}\n")


@dataclass
class FastaRecord:
    """
    Dataclass for a FASTA record.

    Attributes:
        id (str): Sequence identifier.
        description (str): Description of the sequence.
        seq (str): Biological sequence.
    """
    seq_id: str
    description: str
    seq: str

    def __repr__(self) -> str:
        return f"{self.seq_id} {self.description}\n{self.seq}"


class OpenFasta:
    """
    Context manager for reading FASTA files.
    """
    def __init__(self, fasta_file: str, mode: str = 'r'):
        self.file = fasta_file
        self.mode = mode
        self.handler = None
        self.line = None

    def __enter__(self) -> 'OpenFasta':
        self.handler = open(self.file, self.mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.handler.close()

    def __iter__(self) -> Iterator[FastaRecord]:
        return self

    def read_record(self) -> FastaRecord:
        """
        Read the next FASTA record.

        Returns:
            FastaRecord: The next FASTA record.
        """
        return next(self)

    def read_records(self) -> List[FastaRecord]:
        """
        Read all FASTA records.

        Returns:
            Iterator[FastaRecord]: An iterator over all FASTA records.
        """
        records = []
        for record in self:
            records.append(record)
        return records

    def __next__(self) -> FastaRecord:
        if self.line is None:
            self.line = self.handler.readline().strip()
        if self.line == '':
            raise StopIteration

        seq_id, desc = self.line.split(' ', 1)

        seq = ''
        self.line = self.handler.readline().strip()
        while (not self.line.startswith('>')) and (self.line != ''):
            seq = seq + self.line
            self.line = self.handler.readline().strip()

        return FastaRecord(seq_id=seq_id, description=desc, seq=seq)
