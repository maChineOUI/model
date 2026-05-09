#!/usr/bin/env ruby
# frozen_string_literal: true

SOURCE = "report.md"
WORD_OUT = "report_word.md"
TEX_OUT = "report_latex.tex"

def normalize_markdown(lines)
  out = []
  buffer = []
  in_code = false
  in_math_block = false

  flush = lambda do
    next if buffer.empty?

    out << buffer.join(" ").gsub(/[ \t]+/, " ").strip
    buffer.clear
  end

  special = lambda do |raw|
    stripped = raw.strip
    return true if stripped.empty?
    return true if stripped.start_with?("#")
    return true if stripped.start_with?("![")
    return true if stripped == "---"
    return true if stripped.start_with?("|")
    return true if stripped.start_with?("- ", "* ")
    return true if stripped.match?(/^\d+\.\s/)
    return true if raw.end_with?("  ")
    return true if stripped.start_with?("**") && stripped.end_with?("**:")

    false
  end

  lines.each do |line|
    raw = line.chomp
    stripped = raw.strip

    if in_code
      out << raw
      in_code = false if stripped.start_with?("```")
      next
    end

    if in_math_block
      out << raw
      in_math_block = false if stripped == "$$"
      next
    end

    if stripped.start_with?("```")
      flush.call
      out << raw
      in_code = true
      next
    end

    if stripped == "$$"
      flush.call
      out << raw
      in_math_block = true
      next
    end

    if raw.include?("$$")
      flush.call
      out << raw
      next
    end

    if special.call(raw)
      flush.call
      out << raw
      next
    end

    buffer << stripped
  end

  flush.call
  out
end

def split_table_row(line)
  cells = []
  cell = +""
  in_math = false
  i = 0

  while i < line.length
    ch = line[i]
    prev = i.zero? ? nil : line[i - 1]

    if ch == "$" && prev != "\\"
      in_math = !in_math
      cell << ch
    elsif ch == "|" && !in_math
      cells << cell.strip unless i.zero?
      cell = +""
    else
      cell << ch
    end

    i += 1
  end

  cells << cell.strip unless cell.strip.empty?
  cells
end

def escape_text(text)
  text
    .gsub("\\", "\\textbackslash{}")
    .gsub("%", "\\%")
    .gsub("&") { "\\&" }
    .gsub("#", "\\#")
    .gsub("_", "\\_")
end

def convert_inline(text)
  parts = text.split(/(\$[^$]+\$)/)
  converted = parts.map do |part|
    if part.start_with?("$") && part.end_with?("$")
      part
    else
      piece = escape_text(part)
      piece = piece.gsub(/`([^`]+)`/) { "\\texttt{#{$1.gsub(/[\\{}]/) { |m| "\\#{m}" }}}" }
      piece = piece.gsub(/\*\*([^*]+)\*\*/) { "\\textbf{#{$1}}" }
      piece = piece.gsub(/\*([^*]+)\*/) { "\\emph{#{$1}}" }
      piece
    end
  end

  converted.join
end

def extract_title_meta(lines)
  title = ""
  meta = []
  body_start = 0

  lines.each_with_index do |line, idx|
    stripped = line.strip
    if title.empty? && stripped.start_with?("# ")
      title = stripped.sub("# ", "")
    elsif stripped.start_with?("**Course:**", "**Authors:**", "**Date:**", "**Language:**")
      meta << stripped
    elsif stripped == "---"
      body_start = idx + 1
      break
    end
  end

  [title, meta, lines[body_start..]]
end

def convert_display_math(line)
  content = line.strip.sub(/\A\$\$/, "").sub(/\$\$\z/, "").strip
  env = content.include?("\\tag{") ? "equation" : "equation*"
  "\\begin{#{env}}\n#{content}\n\\end{#{env}}"
end

def convert_table(block)
  rows = block.map { |line| split_table_row(line) }
  rows.reject! { |cells| cells.all? { |cell| cell.match?(/\A:?-+:?\z/) } }
  return "" if rows.empty?

  col_count = rows.map(&:length).max
  spec = "|" + (["p{0.18\\textwidth}"] + Array.new(col_count - 1, "p{0.22\\textwidth}")).join("|") + "|"

  tex = []
  tex << "\\begin{center}"
  tex << "\\renewcommand{\\arraystretch}{1.15}"
  tex << "\\begin{tabular}{#{spec}}"
  tex << "\\hline"

  rows.each_with_index do |cells, idx|
    padded = cells + Array.new(col_count - cells.length, "")
    rendered = padded.map { |cell| convert_inline(cell) }
    tex << rendered.join(" & ") + " \\\\"
    tex << "\\hline" if idx.zero? || idx == rows.length - 1
  end

  tex << "\\end{tabular}"
  tex << "\\end{center}"
  tex.join("\n")
end

def convert_body(lines)
  out = []
  paragraph = []
  table_block = []
  list_items = []
  ordered_items = []
  in_code = false
  in_math = false
  math_lines = []

  flush_paragraph = lambda do
    next if paragraph.empty?

    out << convert_inline(paragraph.join(" "))
    out << ""
    paragraph.clear
  end

  flush_table = lambda do
    next if table_block.empty?

    out << convert_table(table_block)
    out << ""
    table_block.clear
  end

  flush_lists = lambda do
    unless list_items.empty?
      out << "\\begin{itemize}"
      list_items.each { |item| out << "\\item #{convert_inline(item)}" }
      out << "\\end{itemize}"
      out << ""
      list_items.clear
    end

    unless ordered_items.empty?
      out << "\\begin{enumerate}"
      ordered_items.each { |item| out << "\\item #{convert_inline(item)}" }
      out << "\\end{enumerate}"
      out << ""
      ordered_items.clear
    end
  end

  lines.each do |line|
    raw = line.chomp
    stripped = raw.strip

    if in_code
      if stripped.start_with?("```")
        out << "\\end{verbatim}"
        out << ""
        in_code = false
      else
        out << raw
      end
      next
    end

    if in_math
      math_lines << raw
      if stripped.end_with?("$$")
        content = math_lines.join("\n").sub(/\A\$\$/, "").sub(/\$\$\z/, "").strip
        env = content.include?("\\tag{") ? "equation" : "equation*"
        out << "\\begin{#{env}}"
        out << content
        out << "\\end{#{env}}"
        out << ""
        in_math = false
        math_lines.clear
      end
      next
    end

    if stripped.start_with?("```")
      flush_paragraph.call
      flush_table.call
      flush_lists.call
      out << "\\begin{verbatim}"
      in_code = true
      next
    end

    if stripped.start_with?("$$") && !stripped.end_with?("$$")
      flush_paragraph.call
      flush_table.call
      flush_lists.call
      in_math = true
      math_lines = [raw]
      next
    end

    if stripped.empty?
      flush_paragraph.call
      flush_table.call
      flush_lists.call
      next
    end

    if stripped.start_with?("|")
      flush_paragraph.call
      flush_lists.call
      table_block << stripped
      next
    else
      flush_table.call
    end

    if stripped.start_with?("- ")
      flush_paragraph.call
      flush_table.call
      flush_lists.call if ordered_items.any?
      list_items << stripped.sub(/\A-\s+/, "")
      next
    end

    if stripped.match?(/^\d+\.\s+/)
      flush_paragraph.call
      flush_table.call
      flush_lists.call if list_items.any?
      ordered_items << stripped.sub(/\A\d+\.\s+/, "")
      next
    end

    if list_items.any?
      list_items[-1] << " " << stripped
      next
    end

    if ordered_items.any?
      ordered_items[-1] << " " << stripped
      next
    end

    flush_lists.call

    if (image_match = stripped.match(/\A!\[(.*?)\]\((.*?)\)\z/))
      caption = image_match[1]
      path = image_match[2]
      out << "\\begin{center}"
      out << "\\includegraphics[width=0.95\\textwidth]{#{path}}"
      out << "\\par\\small #{convert_inline(caption)}" unless caption.empty?
      out << "\\end{center}"
      out << ""
      next
    end

    if stripped.start_with?("## ")
      flush_paragraph.call
      out << "\\section*{#{escape_text(stripped.sub('## ', ''))}}"
      out << ""
      next
    end

    if stripped.start_with?("### ")
      flush_paragraph.call
      out << "\\subsection*{#{convert_inline(stripped.sub('### ', ''))}}"
      out << ""
      next
    end

    if stripped == "---"
      flush_paragraph.call
      out << "\\bigskip"
      out << ""
      next
    end

    if stripped.start_with?("$$") && stripped.end_with?("$$")
      flush_paragraph.call
      out << convert_display_math(stripped)
      out << ""
      next
    end

    paragraph << stripped
  end

  flush_paragraph.call
  flush_table.call
  flush_lists.call
  out
end

source_lines = File.readlines(SOURCE, chomp: false)
word_lines = normalize_markdown(source_lines)
File.write(WORD_OUT, word_lines.join("\n") + "\n")

title, meta_lines, body_lines = extract_title_meta(word_lines)
meta_tex = meta_lines.map do |line|
  convert_inline(line)
    .sub("\\textbf{Course:}", "\\textbf{Course:}")
    .sub("\\textbf{Authors:}", "\\textbf{Authors:}")
    .sub("\\textbf{Date:}", "\\textbf{Date:}")
    .sub("\\textbf{Language:}", "\\textbf{Language:}")
end

body_tex = convert_body(body_lines)

tex = []
tex << "\\documentclass[11pt,a4paper]{article}"
tex << "\\usepackage[margin=2.5cm]{geometry}"
tex << "\\usepackage{amsmath,amssymb,amsthm}"
tex << "\\usepackage[T1]{fontenc}"
tex << "\\usepackage[utf8]{inputenc}"
tex << "\\usepackage{lmodern}"
tex << "\\usepackage{booktabs,array}"
tex << "\\usepackage{graphicx}"
tex << "\\usepackage{hyperref}"
tex << "\\usepackage{verbatim}"
tex << "\\setlength{\\parindent}{0pt}"
tex << "\\setlength{\\parskip}{0.75em}"
tex << "\\begin{document}"
tex << "\\begin{titlepage}"
tex << "\\thispagestyle{empty}"
tex << "\\vspace*{0.6cm}"
tex << "{\\large\\bfseries M1 CHPS\\hfill TP TM --- 2025--2026\\par}"
tex << "\\vspace{0.35cm}"
tex << "\\rule{\\textwidth}{0.6pt}\\par"
tex << "\\vspace{2.6cm}"
tex << "{\\small\\bfseries\\MakeUppercase{Finite Difference Methods}\\par}"
tex << "\\vspace{0.7cm}"
tex << "{\\Huge\\bfseries #{escape_text(title)}\\par}"
tex << "\\vspace{0.8cm}"
tex << "{\\Large Analytical and numerical study of a linear reaction-diffusion equation\\par}"
tex << "\\vfill"
tex << "\\rule{\\textwidth}{0.4pt}\\par"
tex << "\\vspace{0.45cm}"
tex << "\\begin{tabular}{@{}p{0.22\\textwidth}p{0.72\\textwidth}@{}}"
meta_tex.each do |line|
  label, value = line.split("}", 2)
  if value
    clean_label = label.sub("\\textbf{", "").sub(":", "")
    tex << "\\textsc{#{clean_label}} & #{value.strip} \\\\"
  else
    tex << " & #{line} \\\\"
  end
end
tex << "\\end{tabular}\\par"
tex << "\\vspace{0.8cm}"
tex << "{\\small\\hfill Python / NumPy --- Reaction-Diffusion Equation\\par}"
tex << "\\end{titlepage}"
tex.concat(body_tex)
tex << "\\end{document}"

tex_text = tex.join("\n") + "\n"
tex_text.sub!("\\bigskip\n\n\\section*{Abstract}", "\\newpage\n\n\\section*{Abstract}")
tex_text.gsub!("\\end{itemize}\n\n\\begin{itemize}\n", "")
tex_text.gsub!("\\end{enumerate}\n\n\\begin{enumerate}\n", "")
tex_text.gsub!(/\*\*([^*]+)\*\*/, '\\\\textbf{\1}')
tex_text.gsub!("**The reaction term $\\alpha > 0$ is benign**", "\\textbf{The reaction term $\\alpha > 0$ is benign}")

File.write(TEX_OUT, tex_text)
