import pandas as pd
import pylatex as pl

class GlobalReport():
    pass

    def __init__(self):
        """
        Initiate document
        """

        geometry_options = {
            "paper": "a4paper",
            "margin": "1.5cm",
}
        
        self.doc = pl.Document(geometry_options=geometry_options)
        self.doc.preamble.append(pl.Command('title', 'SAMueL Analysis'))
        self.doc.preamble.append(pl.Command('author', 'SAMueL'))
        self.doc.preamble.append(pl.Command('date', ''))
        self.doc.append(pl.NoEscape(r'\maketitle'))


    
    def create_report(self):
        """
        Generate report
        """

        with self.doc.create(pl.Section('Descriptive statistics')):

            # Excample table from dataframe

            df = pd.DataFrame({'a': [1,2,3], 'b': [9,8,7]})
            df.index.name = 'x'

            with self.doc.create(pl.Tabular('ccc')) as table:
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



