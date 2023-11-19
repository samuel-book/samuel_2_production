import pandas as pd
import pylatex as pl


class GlobalReport():

    def __init__(self):
        """
        Initiate document
        """

        geometry_options = {
            "paper": "a4paper",
            "margin": "1.5cm",
        }
    
        self.doc = pl.Document(geometry_options=geometry_options)
        self.doc.packages.append(pl.Package('parskip'))
        self.doc.preamble.append(pl.Command('title', 'SAMueL Analysis'))
        self.doc.preamble.append(pl.Command(
            'author', 'SAMueL Team (contact m.allen@exeter.ac.uk)'))
        self.doc.preamble.append(pl.Command('date', ''))
        self.doc.append(pl.NoEscape(r'\maketitle'))

    def create_report(self):
        """
        Generate report
        """

        with self.doc.create(pl.Section('Descriptive statistics')):

            # Add intro tex from file
            with open('./utils/latex_text/global_ds_intro.txt') as file:
                tex = file.read()
            self.doc.append(pl.NoEscape(tex))
            self.doc.append(pl.Command('vspace', '2mm'))

            # Add general info
            df = pd.read_csv('./output/stats_summary.csv', index_col='field')
            records = df.loc['total records'][0]
            arrival_4hr = df.loc['4 hr arrivals'][0]
            min_year, max_year = df.loc['min year'][0], df.loc['max year'][0]
            txt = (f'The total number of records was {records:,.0f}. ' +
                   f'The year range of the data was {min_year:0.0f}-{max_year:0.0f}. ' +
                   f'The proportion of patients arriving within 4 hours of known onset was {arrival_4hr:0.2f}. ' +
                   'The fraction of each data field that was complete is shown in the table below')
 
            self.doc.append(txt)
            self.doc.append(pl.Command('vspace', '2mm'))

            # Add info on completion
            df = pd.read_csv('./output/full_data_complete.csv', index_col='field')
    

            with self.doc.create(pl.LongTable('l c')) as table:
                table.add_hline()
                table.add_row([df.index.name] + list(df.columns))
                table.add_hline()
                for row in df.index:
                    table.add_row([row] + list(df.loc[row,:]))
                table.add_hline()

        with self.doc.create(pl.Section('Machine learning')):

            self.doc.append('SHAP values for features apart from hospitals are shown in figure 1.')

            with self.doc.create(pl.Figure()) as fig:
                fig.add_image(
                    './../output/thrombolysis_choice_shap_scatter.jpg', 
                    width=("15cm"))
                fig.add_caption('SHAP values')


        self.doc.generate_pdf('./reports/global_report', clean_tex=True)
