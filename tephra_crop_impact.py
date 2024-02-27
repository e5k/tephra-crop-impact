#%%
# Local runs
# import os
# os.chdir('/Users/seb/Library/CloudStorage/OneDrive-unige.ch/Ag-Impact/Streamlit')

# https://coderzcolumn.com/tutorials/data-science/simple-dashboard-using-streamlit
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
from palettable.cartocolors.qualitative import Vivid_10, Prism_4, Prism_10, Vivid_4

# Some config
sns.set_theme(style="whitegrid")
st.set_page_config(page_title='Tephra impact on crops', page_icon=':volcano:',layout="wide")


##% Set mapping dictionaries
cropDict = {
    'cass': 'Cassava',
    'maiz': 'Maize',
    'pota': 'Potato',
    'rice': 'Rice',
    'soyb': 'Soy bean',
    'sugb': 'Sugar beet',
    'whea': 'Wheat',
    'sugc': 'Sugar cane*',
    'oilp': 'Oil palm*',
    'vege': 'Vegetables*'
}
monthDict = {
    0: 'Yearly union',
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December',
}
techDict = {
    'A': 'All',
    'I': 'Irrigated',
    'R': 'Rainfed',
    'H': 'Rainfed, high inputs',
    'L': 'Rainfed, low inputs',
    'S': 'Rainfed, subsistence',
}
dataDict = {
    'A': 'GVP - all VEI',
    'S': 'SE - all VEI',
    'V': 'GVP - VEI occurred',
    'B': 'SE - VEI occurred',
}

months = list(monthDict.keys())
crop = list(cropDict.values())
tech = list(techDict.keys())
# Get list of technologies but without "All" or "Rainfed" --> for pie plots 
techClean = list({k:v for k,v in techDict.items() if k not in ['A','R']}.values()) 

#%% Set Colormaps
# cmapCrop = Vivid_10.hex_colors
cmapCrop = Prism_10.hex_colors
cmapCropd = {crop[i]: cmapCrop[i] for i in range(len(crop))}
# cmapTech = Prism_4.hex_colors
cmapTech = Vivid_4.hex_colors
cmapTechd = {techClean[i]: cmapTech[i] for i in range(len(techClean))}

#%%
## Load Data
@st.cache_data
def load_data():
    df = pd.read_parquet('Data/IMPACT.parquet')
    df[df<0] = 0

    # Scale the production value
    df[['exposedProd', 'impactedProd', 'countryProd']] = df[['exposedProd', 'impactedProd', 'countryProd']]/1e6

    # Format global data
    data = df.reset_index()
    data['crop'] = data['crop'].replace(cropDict)
    data['tech'] = data['tech'].replace(techDict)
    data['ds'] = data['ds'].replace(dataDict)
    # data = data.pivot(index=['country','month', 'VEI', 'tech'],columns='crop', values=['adjProd', 'maxProd', 'countryProd'])
    data = data.pivot(index=['Country','month', 'VEI', 'tech', 'ds', 'thickness'],columns='crop', values=['exposedProd', 'impactedProd', 'countryProd'])

    ## SET DATA
    exposure = data.loc[:,'exposedProd'].drop(index=0, level=1).reset_index() # Exposure for every month of the year
    # exposure['tech'] = exposure['tech'].replace(techDict)
    # exposure['ds'] = exposure['ds'].replace(dataDict)
    exposure.set_index(['ds','Country','tech','VEI','month','thickness'], inplace=True)

    exposure0 = data.loc[:,0,:,:,:,:]['exposedProd'].reset_index() # Exposure for yearly union
    # exposure0['tech'] = exposure0['tech'].replace(techDict)
    # exposure0['ds'] = exposure0['ds'].replace(dataDict)
    exposure0.set_index(['ds','Country','tech','VEI','thickness'], inplace=True)

    loss = data.loc[:,'impactedProd'].drop(index=0, level=1).reset_index()
    # loss['ds'] = loss['ds'].replace(dataDict)
    # loss[crop] = loss[crop]*-1 # Make the losses negative
    loss.set_index(['ds','Country','tech','VEI','month','thickness'], inplace=True)

    prod = data.loc[:,'countryProd'].loc[:,1,5,:,'GVP - all VEI',0.1].reset_index()
    # prod = prod.rename_axis(None, axis=1) # Remove the "crop" column name
    # prod['tech'] = prod['tech'].replace(techDict)
    prod.set_index(['Country','tech'], inplace=True)
    
    return data, exposure, exposure0, loss, prod

data, exposure, exposure0, loss, prod = load_data()

#%%

## Set list values
country = data.reset_index()['Country'].unique()
VEI = [2,3,4,5]
thickness = [0.1,0.5,1]
dataset = ['GVP - all VEI','SE - all VEI','GVP - VEI occurred','SE - VEI occurred']
tech = ['All','Irrigated','Rainfed','Rainfed, high inputs','Rainfed, low inputs','Rainfed, subsistence']

## Sidebar
with st.sidebar:
    dataset_help= """
        #### Eruptive history dataset
        Losses are estimated using two different volcano/eruption datasets:
        - **GVP** refers to the [Global Volcanism Program](https://volcano.si.edu) from the Smithsonian Institution and considers Holocene volcanoes.
        - **SE** refers to the NCEI/WDS [Significant Eruption]((https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.hazards:G10147)) dataset.  A significant eruption is classified as one that meets at least one of the following criteria: caused fatalities, caused moderate damage (approximately $1 million or more), Volcanic Explosivity Index (VEI) of 6 or greater, generated a tsunami, or was associated with a significant earthquake. 
        
        In turn, each dataset considers two approaches to treat the eruption catalogue:
        - **all VEI** considers the isopachs associated with eruptions of VEI 2, 3, 4 and 5 at all volcanoes identified in each database.
        - **VEI occurred** considers the isopachs associated with eruptions that have occurred in the eruptive record of each volcano. 
    """
    
    country_help="""
        #### Country 
        Select the country on which to aggregate crop exposure and impact. Note that the approach considers impacts from tephra fallout originating from neighbouring volcanoes (e.g., impacts in Argentina are dominantly caused by volcanoes located in Chile).
    """

    VEI_help="""
        #### Volcanic Explosivity Index (VEI)
        The **VEI** is a measure of the explosivity of an eruption based on the [volume of tephra fallout](https://en.wikipedia.org/wiki/Volcanic_explosivity_index) introduced by [Newhall and Self (1982)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JC087iC02p01231). Bounds of VEI 2-5 are:
        - **VEI 2**: $10^6$-$10^7\ m^3$  
        - **VEI 3**: $10^7$-$10^8\ m^3$  
        - **VEI 4**: $10^8$-$10^9\ m^3$  
        - **VEI 5**: $10^9$-$10^{10}\ m^3$ 
    """
    
    tech_help="""
        #### Technology
        **Technology** is defined within the [MAPSPAM](https://mapspam.info) dataset and splits crop production by farming practices. 
    """
    
    thick_help="""
        #### Thickness 
        Minimum deposit thickness considered to cause an impact on crop production.
    """
    
    opt_dataset = st.selectbox("Dataset", dataset, help=dataset_help )
    opt_country = st.selectbox("Country", country, 75, help=country_help)
    opt_VEI = st.selectbox("VEI", VEI, index=2, help=VEI_help)
    opt_tech = st.selectbox("Technology", tech, index=0, help=tech_help)
    opt_thickness = st.selectbox("Deposit thickness (cm)", thickness, index=1, help=thick_help)
    # but_update = st.button('Update', on_click=updatePlots)


## Title
st.title("Global crop impact analysis")

"---"

# Widgets State Change Actions & Layout Adjustments.
st.header('Country production')
piePlots = st.container()
piePlot1, _, piePlot2 = piePlots.columns([2,1,2])

with piePlot1:
    st.subheader('Production by crop')
    """
        This pie chart shows the proportion of production for the entire country per crop considering all technologies together.
    """
    prod_crop = pd.DataFrame(prod.loc[opt_country].loc['All']).reset_index()
    prod_crop = prod_crop.rename({'All': 'value', 'index':'crop'},axis=1)
    prod_crop['color'] = prod_crop['crop'].copy()
    prod_crop['color'] = prod_crop['color'].replace(cmapCropd)
    
    fig, ax_piePlot1 = plt.subplots(figsize=(3,3))
    wedges,_ = ax_piePlot1.pie(prod_crop['value'],colors=prod_crop['color'], wedgeprops = {'edgecolor': 'w','linewidth': .5})
    ax_piePlot1.legend(wedges, prod_crop['crop'], ncols=3, loc='center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize='small')
    st.pyplot(fig)
    with st.expander('Production per crop (tons)', expanded=False):
        st.dataframe((prod_crop.drop('color', axis=1).rename({'crop': 'Crop', 'value': 'Production (tons)'},axis=1).set_index('Crop')*1e6).round(), use_container_width=True)
 
with piePlot2:   
    st.subheader('Production by technology')
    """
        This pie chart shows the proportion of production for the entire country per technology aggregating the production of all considered crops.
    """
    prod_tech = pd.DataFrame(prod.loc[opt_country].sum(axis=1).drop(['All', 'Rainfed'])).reset_index().sort_values(by='tech')
    prod_tech = prod_tech.rename({0: 'value'}, axis=1)
    prod_tech['color'] = prod_tech['tech'].copy()
    prod_tech['color'] = prod_tech['color'].replace(cmapTechd)

    fig, ax_piePlot2 = plt.subplots(figsize=(3,3))
    wedges,_ = ax_piePlot2.pie(prod_tech['value'],colors=prod_tech['color'], wedgeprops = {'edgecolor': 'w','linewidth': .5})
    ax_piePlot2.legend(wedges, prod_tech['tech'], ncols=2, loc='center', bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize='small')
    st.pyplot(fig)
    with st.expander('Production per technology (tons)', expanded=False):
        st.dataframe((prod_tech.drop('color', axis=1).rename({'tech': 'Technology', 'value': 'Production (tons)'},axis=1).set_index('Technology')*1e6).round(), use_container_width=True)
    
totProd = st.container()
with totProd:
    st.subheader('Total production')
    """
        This bar chart shows the production of all considered crops by technology. Note that i) *subsistence*, *low input* and *high input* are subsets of *rainfed* and ii) these are **stacked** bars.
    """
    
    fig, ax_totProd = plt.subplots(figsize=(8,2.8))
    prod_plot = prod.loc[opt_country]
    prod_plot = prod_plot.reset_index().sort_values(by='tech').set_index('tech')# .melt(id_vars=['tech'], value_vars=crop, var_name='crop', value_name='value')
    prod_plot = prod_plot[crop]    
    prod_plot.plot(ax=ax_totProd, kind='barh',stacked=True, color=cmapCrop)
    # plt.legend( ncols=4, loc='center', bbox_to_anchor=(0.5, -0.5), frameon=False, fontsize='small')
    plt.legend( loc='right', bbox_to_anchor=(1.3, 0.5), frameon=False, fontsize='small')
    sns.despine(left=True, bottom=True)
    ax_totProd.set(ylabel=None, xlabel='Total production ($\\times$10$^6$ tons)')
    st.pyplot(fig)
    
    with st.expander('Total production data (tons)', expanded=False):
        st.dataframe(prod_plot*1e6, use_container_width=True)


"---"


#%%
st.header('Exposure')
st.subheader('Production exposure')
exp0 = st.container()
exp01, exp02 = exp0.columns([5,1])

with exp01:
    """
        This bar chart shows yearly exposed production estimated by taking the union of the isopach polygon of each separate month.
    """
with exp02:
    scl = st.checkbox('Scale exposure to production', value=True)
with exp0:
    fig, ax_exp0 = plt.subplots(figsize=(8,2.8))
    exp0_plot = exposure0.loc[opt_dataset, opt_country, :, opt_VEI, opt_thickness]
    exp0_plot = exp0_plot.reset_index().sort_values(by='tech').set_index('tech')
    exp0_plot = exp0_plot[crop]    
    exp0_plot.plot(ax=ax_exp0, kind='barh',stacked=True, color=cmapCrop)
    # plt.legend( ncols=4, loc='center', bbox_to_anchor=(0.5, -0.5), frameon=False, fontsize='small')
    plt.legend( loc='right', bbox_to_anchor=(1.3, 0.5), frameon=False, fontsize='small')
    sns.despine(left=True, bottom=True)
    ax_exp0.set(ylabel=None, xlabel='Yearly exposed production ($\\times$10$^6$ tons)')
    
    if scl:
        ax_exp0.set(xlim=ax_totProd.get_xlim())
    st.pyplot(fig)
    with st.expander('Total yearly exposure data (tons)', expanded=False):
        st.dataframe(exp0_plot*1e6, use_container_width=True)
        
"---"
#%%
style_kwargs = {
    # general
    'figure.facecolor': 'w',
    # font sizes
    # 'font.size': 12,
    # 'axes.titlesize': 16,
    # 'xtick.labelsize': 10,
    # force black border
    'patch.force_edgecolor': True,
    'patch.facecolor': 'black',
    # remove spines
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.bottom': False,
    'xtick.top': False,
    'axes.titlepad': 10,
    # grid
    'axes.grid': True,
    # 'grid.color': 'k',
    # 'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    # 'lines.dotted_pattern': [1, 3],
    'lines.scale_dashes': False
}
def butterfly_chart(
        dfe,dfi,clr,ttl,
        middle_label_offset=0.01,
        figsize=(5, 2),
        wspace=0.6
    ):
    
    fig, (ax1, ax2) = plt.subplots(
            figsize=figsize,
            nrows=1,
            ncols=2,
            subplot_kw={'yticks': []},
            gridspec_kw={'wspace': wspace},
        )
           
    # plot the data
    dfi.plot(ax=ax1, kind='barh', stacked=True, color=clr, legend=False, linewidth=.7)
    ax1.invert_xaxis()
    ax1.set_title('Modelled losses')

    dfe.plot(ax=ax2, kind='barh', stacked=True, color=clr, legend=False, linewidth=.7)
    ax2.set_title('Exposure')

    # forced shared xlim
    ax1.set_xlim((ax2.get_xlim()[1],0))

    # turn on axes spines on the inside y-axis
    ax1.spines['right'].set_visible(True)
    ax2.spines['left'].set_visible(True)
    
    # 
    ax1.set_ylabel(None)
    ax2.set_ylabel(None)

    # place center labels
    middle_label_offset=0.01
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                            'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                            'Nov', 'Dec']

    transform = transforms.blended_transform_factory(fig.transFigure, ax1.transData)
    for i, label in enumerate(labels):
        ax1.text(0.5+middle_label_offset, i, label, ha='center', va='center', transform=transform)

    ax1.set_yticklabels('')
    ax2.set_yticklabels('')
    
    # ax1.legend(fontsize=14, facecolor=(1, 1, 1, 0.5), edgecolor=(1,1,1, 0))
    ax2.legend(loc='right', bbox_to_anchor=(1.5, 0.5), facecolor=(1, 1, 1, 0.5), edgecolor=(1,1,1, 0))
    
    fig.supxlabel('Production ($\\times10^6$ tons)',y=-0.05, fontsize=12)
    # fig.suptitle(ttl, fontweight='bold')
    return fig, ax1, ax2


st.header('Impact')
st.subheader('Modelled losses vs production exposure')
"""
    This tornado chart compares modelled losses vs production exposure for each month.
"""
# Subset and get columns
loss_plot = loss.loc[opt_dataset, opt_country, opt_tech, opt_VEI, :, opt_thickness][crop]
exposure_plot = exposure.loc[opt_dataset, opt_country, opt_tech, opt_VEI, :, opt_thickness][crop]
# Indonesia colored by crop
with matplotlib.rc_context(style_kwargs):
    fig, ax1, ax2 = butterfly_chart(
        exposure_plot, loss_plot, cmapCrop,
        None,
        figsize=(10, 5),
        wspace=0.35,
        middle_label_offset=0.015,
    )
    st.pyplot(fig)

with st.expander('Monthly exposure  (tons)', expanded=False):
        st.dataframe(exposure_plot*1e6, use_container_width=True)
with st.expander('Monthly modelled losses  (tons)', expanded=False):
        st.dataframe(loss_plot*1e6, use_container_width=True)