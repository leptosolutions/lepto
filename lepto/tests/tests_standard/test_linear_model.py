from importlib.resources import files
import pandas as pd
import numpy as np
from lepto.standard.model.linear_model import GLMDiff

class TestGLMDiff:
    def test_model(self):
        path = files("lepto.data") / "sample_standard.csv"
        df = pd.read_csv(path)

        glm = GLMDiff(
            family="poisson",      
            tweedie_power=1.5,       
            fit_intercept=True,
            nbins=10,                
            lam=100,
            penalty_choice=None,
            offset_betas=None)
        
        glm.fit(
            X=df[["age", "region", "segment"]],
            y=df["y"].values,
            sample_weight=df["exposure"].values
        )
        pred = glm.predict(df[["age", "region", "segment"]])
        summary_glm = glm.summary

        assert summary_glm is not None
        assert pred is not None

    def test_model_penalty_continuous(self):
        path = files("lepto.data") / "sample_standard.csv"
        df = pd.read_csv(path)

        k = len(df['region'].unique())
        adj = np.zeros((k, k))
        for i in range(k):
            adj[i, (i+1) % k] = 1.0
            adj[i, (i-1) % k] = 1.0

        penalty = {'age': {'penalty':'continuous'}}

        glm = GLMDiff(
            family="poisson",      
            tweedie_power=1.5,       
            fit_intercept=True,
            nbins=10,                
            lam=1e10,
            penalty_choice=penalty,
            offset_betas=None)
        
        glm.fit(
            X=df[["age", "region", "segment"]],
            y=df["y"].values,
            sample_weight=df["exposure"].values
        )
        summary_glm = glm.summary

        assert np.allclose(list(summary_glm['coefficients']['age'].values()), 0.0, atol=1e-4)

    def test_model_penalty_adj(self):
        path = files("lepto.data") / "sample_standard.csv"
        df = pd.read_csv(path)

        k = len(df['region'].unique())
        adj = np.zeros((k, k))
        adj[1, 2] = 1
        adj[2, 1] = 1

        penalty = {'region': {'penalty':'graph', 'graph':adj}}

        glm = GLMDiff(
            family="poisson",      
            tweedie_power=1.5,       
            fit_intercept=True,
            nbins=10,                
            lam=1e10,
            penalty_choice=penalty,
            offset_betas=None)
        
        glm.fit(
            X=df[["age", "region", "segment"]],
            y=df["y"].values,
            sample_weight=df["exposure"].values
        )
        summary_glm = glm.summary

        assert np.isclose(summary_glm['coefficients']['region']['south'], summary_glm['coefficients']['region']['south'])

    def test_model_offset(self):
        path = files("lepto.data") / "sample_standard.csv"
        df = pd.read_csv(path)

        offset_betas = np.array([ np.nan  ,  np.nan   ,  np.nan   ,  np.nan   ,  np.nan   ,
        np.nan  ,  np.nan  ,  np.nan  ,  np.nan   , np.nan   ,
        1.5   ,  np.nan   ,  np.nan   , np.nan])

        glm = GLMDiff(
            family="poisson",      
            tweedie_power=1.5,       
            fit_intercept=True,
            nbins=10,                
            lam=1e10,
            penalty_choice=None,
            offset_betas=offset_betas)
        
        glm.fit(
            X=df[["age", "region", "segment"]],
            y=df["y"].values,
            sample_weight=df["exposure"].values
        )
        summary_glm = glm.summary
        assert np.isclose(summary_glm['coefficients']['region']['north'], 1.5)

    def test_model_categories(self):
        path = files("lepto.data") / "sample_standard.csv"
        df = pd.read_csv(path)

        categories = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [0.0, 1.0], ['east', 'north', 'south', 'west', 'new_zone']]

        glm = GLMDiff(
            family="poisson",      
            tweedie_power=1.5,       
            fit_intercept=True,
            nbins=10,                
            lam=1e10,
            penalty_choice=None,
            offset_betas=None,
            categories=categories)
        
        glm.fit(
            X=df[["age", "region", "segment"]],
            y=df["y"].values,
            sample_weight=df["exposure"].values
        )
        expected = [np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=object),
                    np.array([0.0, 1.0], dtype=object),
                    np.array(['east', 'north', 'south', 'west', 'new_zone'], dtype=object)]
        result = glm.data.transformer.named_steps['onehot'].categories_
        for x, y in zip(expected, result):
            assert np.array_equal(x, y)
