from flask import Flask, request, render_template, send_file
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import base64

app = Flask(__name__)

# Initialize an empty dataframe and history stack
df = pd.DataFrame()
history = []

def handle_missing_values(df, column):
    if column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df

def encode_categorical_columns(df, column):
    if column in df.columns and df[column].dtype in ['object', 'category']:
        df = pd.get_dummies(df, columns=[column], drop_first=True)
    return df

def drop_duplicates(df, column):
    if column in df.columns:
        df.drop_duplicates(subset=[column], inplace=True)
    return df

def scale_numeric_column(df, column):
    if column in df.columns and df[column].dtype in ['float64', 'int64']:
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[[column]])
    return df

def delete_column(df, column):
    if column in df.columns:
        df.drop(columns=[column], inplace=True)
    return df

def get_column_statistics(df, column):
    stats = {}
    if column in df.columns:
        stats['Unique Values Count'] = df[column].nunique()
        stats['Null Values Count'] = df[column].isnull().sum()
        stats['Unique Values'] = df[column].unique().tolist()

        if df[column].dtype in ['float64', 'int64']:
            stats['Max'] = df[column].max()
            stats['Min'] = df[column].min()
            stats['Mean'] = df[column].mean()
            stats['Standard Deviation'] = df[column].std()
            stats['Variance'] = df[column].var()
            stats['Median'] = df[column].median()
            stats['Mode'] = df[column].mode()[0]
        else:
            stats['Mode'] = df[column].mode()[0]
            stats['Top 5 Frequent Values'] = df[column].value_counts().head().to_dict()
    return stats

def calculate_dataset_details(df):
    num_rows = len(df)
    num_columns = len(df.columns)
    num_numerical = len(df.select_dtypes(include=['number']).columns)
    num_categorical = len(df.select_dtypes(include=['object']).columns)
    column_names = df.columns.tolist()
    return {
        'num_rows': num_rows,
        'num_columns': num_columns,
        'num_numerical': num_numerical,
        'num_categorical': num_categorical,
        'column_names': column_names
    }


def generate_plot(df, plot_type, x_column=None, y_column=None):
    buffer = BytesIO()
    plt.figure(figsize=(10, 6))
    if plot_type == 'histogram' and x_column:
        sns.histplot(df[x_column].dropna(), kde=True)
    elif plot_type == 'scatter' and x_column and y_column:
        sns.scatterplot(data=df, x=x_column, y=y_column)
    elif plot_type == 'box' and x_column:
        sns.boxplot(x=df[x_column])
    plt.title(f'{plot_type.capitalize()} of {x_column} vs {y_column}' if y_column else f'{plot_type.capitalize()} of {x_column}')
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def undo_last_operation():
    global df, history
    if len(history) > 1:
        history.pop()  # Remove the last state
        df = history[-1].copy()  # Restore the previous state

@app.route('/', methods=['GET', 'POST'])
def index():
    global df, history
    column_stats = {}
    column = ''  # Initialize column variable
    plot_image = None
    dataset_details = None

    if request.method == 'POST':
        if 'upload_file' in request.files:
            file = request.files['upload_file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                history = [df.copy()]  # Initialize history stack
        
        if 'handle_missing_values' in request.form:
            column = request.form['column']
            history.append(df.copy())  # Save the current state
            df = handle_missing_values(df, column)
        
        if 'encode_categorical_columns' in request.form:
            column = request.form['column']
            history.append(df.copy())  # Save the current state
            df = encode_categorical_columns(df, column)
        
        if 'drop_duplicates' in request.form:
            column = request.form['column']
            history.append(df.copy())  # Save the current state
            df = drop_duplicates(df, column)
        
        if 'scale_numeric_column' in request.form:
            column = request.form['column']
            history.append(df.copy())  # Save the current state
            df = scale_numeric_column(df, column)
        
        if 'delete_column' in request.form:
            column = request.form['column']
            history.append(df.copy())  # Save the current state
            df = delete_column(df, column)
        
        if 'undo' in request.form:
            undo_last_operation()
        
        if 'export_csv' in request.form:
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name='processed_dataset.csv', mimetype='text/csv')

        if 'show_column_statistics' in request.form:
            column = request.form['column']
            column_stats = get_column_statistics(df, column) if column and not df.empty else {}

        if 'show_dataset_details' in request.form:
            dataset_details = calculate_dataset_details(df) if not df.empty else {}

        if 'plot_type' in request.form:
            plot_type = request.form['plot_type']
            x_column = request.form['x_column'] if 'x_column' in request.form else None
            y_column = request.form['y_column'] if 'y_column' in request.form else None
            if x_column in df.columns and (plot_type != 'scatter' or (x_column in df.columns and y_column in df.columns)):
                plot_image = generate_plot(df, plot_type, x_column, y_column)

    columns = df.columns.tolist() if not df.empty else []
    sample_size = min(10, len(df)) if not df.empty else 0
    preview_df = df.sample(n=sample_size) if sample_size > 0 else pd.DataFrame()

    return render_template('index.html',
                           preview=preview_df.to_html(classes='table table-striped', index=False),
                           columns=columns,
                           column_stats=column_stats,
                           column_name=column,
                           plot_image=plot_image,
                           dataset_details=dataset_details)

if __name__ == '__main__':
    app.run(debug=True)
