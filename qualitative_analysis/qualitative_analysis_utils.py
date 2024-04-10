import sys
from pathlib import Path

sys.path.insert(0, str(Path("../").resolve()))
import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np
import matplotlib.pyplot as plt
from evaluate_retrieval import is_successful_retrieval

def calculate_success_percentage_qa(data, retrieval_key):
    hop_counts = {2: {'total': 0, 'successful': 0},
                    3: {'total': 0, 'successful': 0},
                    4: {'total': 0, 'successful': 0}}


    for hop_count in data:
        int_hop_count = int(hop_count)
        for key in data[hop_count]:
            for item in data[hop_count][key]:
                #if obj["label"] == "SUPPORTED":
                hop_counts[int_hop_count]["total"] += 1
                if is_successful_retrieval(item, retrieval_key=retrieval_key):
                    hop_counts[int_hop_count]['successful'] += 1

    success_percentages = {}
    for num_hops, counts in hop_counts.items():
        if counts['total'] > 0:
            success_percentage = (counts['successful'] / counts['total']) * 100
            success_percentages[num_hops] = success_percentage


    average_total_percentage = (success_percentages[2] + success_percentages[3] + success_percentages[4]) / 3
    # Example variables
    hops_2 = round(success_percentages[2],2)
    hops_3 = round(success_percentages[3],2)
    hops_4 = round(success_percentages[4],2)
    avg_total = round(average_total_percentage,2)
    return hops_2, hops_3, hops_4, avg_total

def plot_retrieval_success(df, plot_save_path, show_plot=False):
  df_vis = df[["Hop Count", "Supported", "Successful Retrieval"]]
  summary = df_vis.groupby(['Hop Count', 'Supported', 'Successful Retrieval']).size().unstack(fill_value=0).reset_index()

  supported_data = summary[summary['Supported'] == 'SUPPORTED']
  not_supported_data = summary[summary['Supported'] == 'NOT_SUPPORTED']

  fig, ax = plt.subplots(figsize=(10, 6))
  bar_width = 0.35
  index = np.arange(len(supported_data['Hop Count'].unique()))
  bar_positions_supported = index - bar_width / 2
  bar_positions_not_supported = index + bar_width / 2

  ax.bar(bar_positions_supported, supported_data[True], width=bar_width, label='SUPPORTED - True', color='lightgreen')
  ax.bar(bar_positions_supported, supported_data[False], bottom=supported_data[True], width=bar_width, label='SUPPORTED - False', color='darkgreen')
  ax.bar(bar_positions_not_supported, not_supported_data[True], width=bar_width, label='NOT_SUPPORTED - True', color='lightcoral')
  ax.bar(bar_positions_not_supported, not_supported_data[False], bottom=not_supported_data[True], width=bar_width, label='NOT_SUPPORTED - False', color='darkred')

  ax.set_xlabel('Hop Count')
  ax.set_ylabel('Count')
  ax.set_title('Counts by Hop Count, Support Status, and Success')
  ax.set_xticks(index)
  ax.set_xticklabels(supported_data['Hop Count'].unique())
  ax.legend()

  plt.tight_layout()
  if plot_save_path:
    plt.savefig(plot_save_path)
  if show_plot:
    plt.show()

def create_html_report(df, two_hop_accuracy, three_hop_accuracy, four_hop_accuracy, average_total_accuracy, encoded_string, html_save_path, escaped_prompt_template, decomposed_claims_key):
  image_html = f'<img src="data:image/png;base64,{encoded_string}" alt="Chart" style="max-width:100%;height:auto;">'

  html_content = """
  <!DOCTYPE html>
  <html lang="en">
  <head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Retrieval Data Analysis</title>
  <style>
    body {font-family: Arial, sans-serif;}
    .expandable-content {margin-bottom: 10px;}
    .expandable-content .content {display: none;}
    .expandable-content .toggle-button {cursor: pointer; color: #007bff; text-decoration: underline;}
    img {max-width: 100%; height: auto;}
    table {width: 100%; border-collapse: collapse; table-layout: fixed;}
    th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}
    th {background-color: #f2f2f2; position: sticky; top: 0; z-index: 10;}
    .table-container {max-height: 400px; overflow-y: auto;}
  </style>
  <script>
  function toggleContent(id) {
    var content = document.getElementById('content-' + id);
    var button = document.getElementById('button-' + id);
    if (content.style.display === 'none') {
      content.style.display = 'block';
      button.textContent = 'Collapse';
    } else {
      content.style.display = 'none';
      button.textContent = 'Expand';
    }
  }
  </script>
  </head>
  <body>

  <h2>Data Analysis</h2>
  <table>
    <tr>
      <th style="width: 50px;">Index</th>
      <th>Claim</th>
      <th>Decomposed Claims</th>
      <th>Supporting Facts</th>
      <th>Found</th>
      <th>Not Found</th>
      <th style="width: 50px;">Hop Count</th>
      <th>Supported</th>
      <th style="width: 100px;">Successful Retrieval</th>
      <th>Retrieved Documents</th>
    </tr>
  """

  for index, row in df.iterrows():
      html_content += f"""
    <tr>
      <td>{index + 1}</td>
      <td>{row['claim']}</td>
      <td>{row[decomposed_claims_key]}</td>
      <td>{row['supporting_facts']}</td>
      <td>{row['found']}</td>
      <td>{row['not_found']}</td>
      <td>{row['Hop Count']}</td>
      <td>{row['Supported']}</td>
      <td>{row['Successful Retrieval']}</td>
      <td>
        <div class="expandable-content">
          <span id="button-{index}" class="toggle-button" onclick="toggleContent({index});">Expand</span>
          <div id="content-{index}" class="content">{row['Retrieved Documents']}</div>
        </div>
      </td>
    </tr>
  """


  html_content += """
  </table>

  <div style="display: flex; justify-content: space-around; align-items: flex-start;">
    <div style="flex: 1;">
      <h2>Chart</h2>
  """
  html_content += image_html
  html_content += f"""
    </div>

    <div style="flex: 1;">
      <h2>Accuracy Percentages</h2>
      <ul>
        <li>2 Hop Accuracy: {two_hop_accuracy}%</li>
        <li>3 Hop Accuracy: {three_hop_accuracy}%</li>
        <li>4 Hop Accuracy: {four_hop_accuracy}%</li>
        <li>Total Average Accuracy: {average_total_accuracy}%</li>
      </ul>
    </div>
  </div>
  """

  html_content += f"""
  <h2>Prompt Template</h2>
  <pre>
    {escaped_prompt_template}
  </pre>

  """

  html_content += """
  </body>
  </html>
  """


  with open(html_save_path, 'w') as file:
      file.write(html_content)