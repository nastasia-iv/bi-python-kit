from typing import Tuple


def convert_multiline_fasta_to_oneline(input_fasta: str, output_fasta: str = '') -> None:
    """
    Converts multi-line FASTA sequences to single-line sequences

    Reads a FASTA file in which some sequences are written on several lines,
    and writes it to a new file, where each sequence is written on one line.

    Arguments:
        input_fasta (str): path to the FASTA file to read
        output_fasta (str, optional): name for the FASTA file. If not specified, a file with the input name and postfix '_oneline' is created

    Returns:
        None
    """
    if output_fasta == '':  # Если имя output не задано, используем имя input c _oneline
        output_fasta = input_fasta.replace('.fasta', '_oneline.fasta')
    else:  # Если имя output задано, проверим расширение
        if not output_fasta.endswith(".fasta"):
            output_fasta += ".fasta"
    output_lines = []  # Финальный список
    with open(input_fasta, mode="r") as input_file:
        current_sequence = []  # Временный список для строк текущей последовательности
        for line in input_file:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence:  # Если временный список не пуст
                    output_lines.append(
                        "".join(current_sequence))  # Сливаем все последовательности в одну в финальный список
                output_lines.append(line)  # Добавляем идентификатор в финальный список
                current_sequence = []  # Очищаем временный список
            else:
                current_sequence.append(line)  # Добавляем строку во временный список
        # После блока for финальный список заканчивается на последнем идентификаторе, а все его последовательности остались во временном списке
        # Дописываем последовательности для последнего идентификатора из временного списка в финальный
        if current_sequence:
            output_lines.append("".join(current_sequence))
    with open(output_fasta, mode="w") as output_file:
        output_file.write("\n".join(
            output_lines))  # Записываем строки финального списка в output, разделяя каждую символом новой строки


def select_genes_from_gbk_to_fasta(input_gbk: str, genes: Tuple[str], n_before: int = 1, n_after: int = 1,
                                   output_fasta: str = '') -> None:
    """
    Selects neighbor genes for the gene of interest from the GBK file and writes their protein sequences into FASTA format.

     Arguments:
         input_gbk (str): path to the GBK file to read
         genes (Tuple[str]): list of genes of interest
         n_before (int, optional): number of genes before gene of interest. Defaults to 1
         n_after (int, optional): number of genes after gene of interest. Defaults to 1
         output_fasta (str, optional): path to the FASTA file to write to. If not specified, a file with the input name and postfix '_selected' is created

     Returns:
         None
     """
    if output_fasta == '':  # Задаем имя выходному файлу, если оно не задано пользователем
        output_fasta = input_gbk.replace(".gbk", "_selected.fasta")
    else:  # Если имя output задано, проверим расширение
        if not output_fasta.endswith(".fasta"):
            output_fasta += ".fasta"

            # Блок для создания списка генов и словаря
    gene_protein_dict = {}  # Создаем пустой словарь для хранения информации о генах
    with open(input_gbk, mode="r") as input_file:
        current_gene = None
        current_translation = None
        in_gene = False
        gene_list = []
        for line in input_file:
            line = line.strip()
            if line.startswith("/gene="):  # Ищем название гена
                current_gene = line.split("=")[1].strip('"\n')
                line = line.replace('/gene="', '')
                line = line.strip('"')
                gene_list.append(line)  # Добавляем чистое имя гена в список генов
                in_gene = True
            elif in_gene and line.startswith("/translation="):  # Если нашли ген, ищем белковую последовательность
                current_translation = line.split("=")[1].strip(
                    '"')  # Разделитель =, берем только элемент 1, откусываем " в строке
                while not line.endswith('"'):  # Пока не найден конец последовательности
                    line = input_file.readline().strip()  # Читаем следующую строку
                    current_translation += line.strip('"')
                in_gene = False  # Сбрасываем отметку гена

                if current_gene and current_translation:  # Если нашли и название гена, и его белковую последовательность, добавляем их в словарь
                    gene_protein_dict[current_gene] = current_translation
                    current_gene = None
                    current_translation = None

    # Блок для поиска белковых последовательностей генов-соседей
    neighbor_gene_proteins = []  # Создаем список для хранения списка кортежей (ген-сосед, последовательность)
    for gene in genes:  # Итерируемся по генам интереса
        if any(gene in name for name in gene_protein_dict):
            index = gene_list.index(gene)  # Находим индекс гена интереса в общем списке
            # Вычисляем индексы соседних генов
            start = max(0, index - n_before)  # Не брать отрицательные значения
            end = min(len(gene_list), index + n_after + 1)  # Не брать значения > длины списка генов
            # Итерируемся по индексам соседних генов
            for index in range(start, end):
                neighbor_gene = gene_list[index]  # Находим по индексам нужные гены-соседи
                if neighbor_gene != gene:  # Не берём сам ген интереса
                    neighbor_gene_proteins.append((neighbor_gene, gene_protein_dict[neighbor_gene]))  # Записываем в финальный список ген-сосед, последовательность

    # Записываем результат в нужном формате
    with open(output_fasta, mode="w") as output_file:
        for gene_and_translation in neighbor_gene_proteins:
            gene, translation = gene_and_translation  # Распаковываем список в кортежи
            output_file.write(f">{gene}\n{translation}\n")
