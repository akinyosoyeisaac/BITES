import numpy as np
import streamlit as st
from bites.analyse.analyse_utils import *
from data.RGBSG.RGBSG_utilis import load_RGBSG, load_RGBSG_no_onehot
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def main():
    """Choose Method to analyse!"""
    method = 'BITES'
    compare_against_ATE = False

    X_train, Y_train, event_train, treatment_train, _, _ = load_RGBSG(partition='train',
                                                                      filename_="data/RGBSG/rgbsg.h5")
    X_test, Y_test, event_test, treatment_test, _, _ = load_RGBSG(partition='test',
                                                                  filename_="data/RGBSG/rgbsg.h5")

    if method == 'BITES' or method == 'ITES':
        model, config = get_best_model('example_results/' + method + "_RGBSG")
        model.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])

        C_index, _, _ = get_C_Index_BITES(model, X_test, Y_test, event_test, treatment_test)
        pred_ite, _ = get_ITE_BITES(model, X_test, treatment_test)

        if compare_against_ATE:
            sns.set(font_scale=1)
            analyse_randomized_test_set(np.ones_like(pred_ite), Y_test, event_test, treatment_test, C_index=C_index,
                                        method_name=None, save_path=None, annotate=False)
            analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=C_index,
                                        method_name=method,
                                        save_path=None,
                                        new_figure=False, annotate=True)
        else:
            analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=C_index,
                                        method_name=method,
                                        save_path=None)
    elif method == 'DeepSurvT':
        model0, config0 = get_best_model("ray_results/" + method + "_T0_RGBSG", assign_treatment=0)
        model0.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])

        model1, config1 = get_best_model("ray_results/" + method + "_T1_RGBSG", assign_treatment=1)
        model1.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])

        C_index, _, _ = get_C_Index_DeepSurvT(model0, model1, X_test, Y_test, event_test, treatment_test)

        pred_ite, _ = get_ITE_DeepSurvT(model0, model1, X_test, treatment_test, best_treatment=None,
                                        death_probability=0.5)

        # sns.set(font_scale=1)
        analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=C_index,
                                    method_name='T-DeepSurv',
                                    save_path='RGBSG_' + method + '.pdf')

    do_SHAP = True
    if do_SHAP and method == 'BITES':

        model.eval()

        def net_treatment0(input):
            ohc = OneHotEncoder(sparse=False)
            X_ohc = ohc.fit_transform(input[:, -2:])
            tmp = np.c_[input[:, :-2], X_ohc].astype('float32')
            return model.risk_nets[0](model.shared_net(torch.tensor(tmp))).detach().numpy()

        def net_treatment1(input):
            ohc = OneHotEncoder(sparse=False)
            X_ohc = ohc.fit_transform(input[:, -2:])
            tmp = np.c_[input[:, :-2], X_ohc].astype('float32')
            return model.risk_nets[1](model.shared_net(torch.tensor(tmp))).detach().numpy()

        #Load data without one_hot encoding

        X_train, Y_train, event_train, treatment_train, _ = load_RGBSG_no_onehot(partition='train',
                                                                                 filename_="data/RGBSG/rgbsg.h5")

        X_test, Y_test, event_test, treatment_test, _ = load_RGBSG_no_onehot(partition='test',
                                                                             filename_="data/RGBSG/rgbsg.h5")


        st.markdown(r"<h1 style='text-align:center'> INTERACTIVE VISUAL OF THE BITES' ALGORITHM </h1>", unsafe_allow_html=True)
        # temp = open("order.plk", "rb")
        # order = pickle.load(temp)
        names = ['N pos nodes', 'Age', 'Progesterone', 'Estrogene', 'Menopause', 'Grade']
        #features = st.selectbox(label="select feature of interest", options=names)
        temp = st.selectbox(label="select temp of interest", options=["shap_values0_temp", "shap_values1_temp"])
        plots = st.selectbox(label="select plot of interest", options=["beeswarm", "force_plot"])
        

        names = ['N pos nodes', 'Age', 'Progesterone', 'Estrogene', 'Menopause', 'Grade']

        X_train0 = X_train[treatment_train == 0]
        X_test0 = pd.DataFrame(X_test[treatment_test == 0], columns=names)
        explainer_treatment0 = shap.Explainer(net_treatment0, X_train0)
        

        X_train1 = X_train[treatment_train == 1]
        X_test1 = pd.DataFrame(X_test[treatment_test == 1], columns=names)
        explainer_treatment1 = shap.Explainer(net_treatment1, X_train1)
        



        if temp == "shap_values0_temp" and plots == "beeswarm":
            instance0 = st.number_input(label="Enter an instance for the treatment0", min_value=int(X_test0.index.min()),
                                       max_value=int(X_test0.index.max()), value=int(X_test0.index.max() / 2), step=10,
                                       key='treatment0')
            shap_values0_temp = explainer_treatment0(X_test0.iloc[instance0].to_frame().T.astype('float32'))
            plt.style.use('default')
            # dict_of_fectures = dict(enumerate(names))
            # dict_of_fectures = {x: y for x, y in zip(dict_of_fectures.values(), dict_of_fectures.keys())}
            # plt.axes(ax[0])
            # fig, ax = plt.subplots()
            fig = plt.figure()
            # shap.plots.beeswarm(shap_values0_temp, color_bar_label=None, color_bar=None, order=order.abs.mean(0))
            shap.plots.beeswarm(shap_values0_temp, order=shap_values0_temp.abs.mean(0), 
                                color_bar_label=None, color_bar=None)
            plt.xlabel('SHAP value')
            plt.ylabel(f"instance {instance0}")
            plt.xlim([-0.5, 1])
            plt.annotate('a', xy=(0.02, 0.92), xycoords='axes fraction', fontsize='x-large')
            fig.tight_layout()
            plt.title('No Hormone Treatment')
            st.pyplot(fig=fig)

        elif temp == "shap_values1_temp" and plots == "beeswarm":
            instance1 = st.number_input(label="Enter an instance for the treatment1", min_value=int(X_test1.index.min()),
                                       max_value=int(X_test1.index.max()), value=int(X_test1.index.max() / 2), step=10,
                                       key='treatment1')
            shap_values1_temp = explainer_treatment1(X_test1.iloc[instance1].to_frame().T.astype('float32'))
            plt.style.use('default')
            # fig, ax = plt.subplots()
            # plt.axes(ax[1])
            fig = plt.figure()
            # shap.plots.beeswarm(shap_values1_temp, order=order.abs.mean(0))
            shap.plots.beeswarm(shap_values1_temp, order=shap_values1_temp.abs.mean(0),
                                color_bar_label=None, color_bar=None)
            plt.title('Hormone Treatment')
            plt.xlabel('SHAP value')
            plt.ylabel(f"instance {instance1}")
            # ax.set_yticks([])
            plt.xlim([-0.5, 1])
            plt.annotate('b', xy=(0.02, 0.92), xycoords='axes fraction', fontsize='x-large')
            fig.tight_layout()
            st.pyplot(fig)
            # plt.savefig('SHAP_RGBSG_BITES.pdf', format='pdf')

        if temp == "shap_values0_temp" and plots == "force_plot":
            instance0 = st.number_input(label="Enter an instance for the treatment0", min_value=int(X_test0.index.min()),
                                       max_value=int(X_test0.index.max()), value=int(X_test0.index.max() / 2), step=10,
                                       key='treatment0')
            shap_values0_temp = explainer_treatment0(X_test0.iloc[instance0].to_frame().T.astype('float32'))
            # plt.style.use('default')
            # plt.axes(ax[0])
            # fig, ax = plt.subplots()
            #fig = plt.figure()
            # shap.plots.beeswarm(shap_values0_temp, color_bar_label=None, color_bar=None, order=order.abs.mean(0))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig = shap.plots.force(base_value=shap_values0_temp.base_values.flatten(), shap_values=shap_values0_temp.values, 
                                    matplotlib=True, feature_names=X_test0.columns.to_list())
            st.pyplot(fig=fig)
            # plt.xlabel('SHAP value')
            # plt.xlim([-0.5, 1])
            # plt.annotate('a', xy=(0.02, 0.92), xycoords='axes fraction', fontsize='x-large')
            # fig.tight_layout()
            # plt.title('No Hormone Treatment')
            

        elif temp == "shap_values1_temp" and plots == "force_plot":
            instance1 = st.number_input(label="Enter an instance for the treatment1", min_value=int(X_test1.index.min()),
                                       max_value=int(X_test1.index.max()), value=int(X_test1.index.max() / 2), step=10,
                                       key='treatment1')
            shap_values1_temp = explainer_treatment1(X_test1.iloc[instance1].to_frame().T.astype('float32'))
            # plt.style.use('default')
            # fig, ax = plt.subplots()
            # plt.axes(ax[1])
            # fig = plt.figure()
            # shap.plots.beeswarm(shap_values1_temp, order=order.abs.mean(0))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig = shap.plots.force(base_value=shap_values1_temp.base_values.flatten(), shap_values=shap_values1_temp.values, 
                                    matplotlib=True, feature_names=X_test1.columns.to_list())
            st.pyplot(fig=fig)
            # plt.title('Hormone Treatment')
            # plt.xlabel('SHAP value')
            # ax.set_yticks([])
            # plt.xlim([-0.5, 1])
            # plt.annotate('b', xy=(0.02, 0.92), xycoords='axes fraction', fontsize='x-large')
            # fig.tight_layout()
            # st.pyplot(fig)
            # plt.savefig('SHAP_RGBSG_BITES.pdf', format='pdf')


if __name__ == '__main__':
    main()
