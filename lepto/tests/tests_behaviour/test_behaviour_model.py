from importlib.resources import files
import pandas as pd
import numpy as np
from lepto.behaviour.model.linear_demand import GLMDemand

class TestGLMDemand:
    def test_model(self):
        path = files("lepto.data") / "sample_behaviour.csv"
        df = pd.read_csv(path)

        X = df[["age", "income", "region", "promo", "loyalty", "price"]]
        w = df['w']
        y = df['y']

        glm = GLMDemand(
            var_glm_static=["age", "income", "region"],
            var_glm_elasticity=["promo", "loyalty"],
            var_behaviour="price",
            direction="increasing",
            static_penalty_choice=None,
            elasticity_penalty_choice=None,
            nbins=10,
            lam=100,
            lam_behaviour=1e5
        )

        glm.fit(X, y, sample_weight=w)
        pred = glm.predict(X)
        summary_glm = glm.summary

        assert summary_glm is not None
        assert pred is not None

    def test_model_penalty_static(self):
        path = files("lepto.data") / "sample_behaviour.csv"
        df = pd.read_csv(path)

        X = df[["age", "income", "region", "promo", "loyalty", "price"]]
        w = df['w']
        y = df['y']

        penalty = {'age': {'penalty':'continuous'}}

        glm = GLMDemand(
            var_glm_static=["age", "income", "region"],
            var_glm_elasticity=["promo", "loyalty"],
            var_behaviour="price",
            direction="increasing",
            static_penalty_choice=penalty,
            elasticity_penalty_choice=None,
            nbins=10,
            lam=1e20,
            lam_behaviour=1e5
        )

        glm.fit(X, y, sample_weight=w)
        summary_glm = glm.summary

        assert np.allclose(list(summary_glm['static']['coefficients']['age'].values()), 0.0, atol=1e-4)

    def test_model_penalty_dynamic(self):
        path = files("lepto.data") / "sample_behaviour.csv"
        df = pd.read_csv(path)

        X = df[["age", "income", "region", "promo", "loyalty", "price"]]
        w = df['w']
        y = df['y']

        penalty = {'promo': {'penalty':'continuous'}}

        glm = GLMDemand(
            var_glm_static=["age", "income", "region"],
            var_glm_elasticity=["promo", "loyalty"],
            var_behaviour="price",
            direction="increasing",
            static_penalty_choice=None,
            elasticity_penalty_choice=penalty,
            nbins=10,
            lam=1e20,
            lam_behaviour=1e5
        )

        glm.fit(X, y, sample_weight=w)
        summary_glm = glm.summary

        assert np.allclose(list(summary_glm['elasticity']['coefficients']['promo'].values()), 0.0, atol=1e-4)

    def test_model_offset(self):
        path = files("lepto.data") / "sample_behaviour.csv"
        df = pd.read_csv(path)

        X = df[["age", "income", "region", "promo", "loyalty", "price"]]
        w = df['w']
        y = df['y']

        offset_static = np.array([np.nan, np.nan,  np.nan,  np.nan,  np.nan,
        np.nan,  np.nan, np.nan,  np.nan, np.nan,
       np.nan, np.nan, np.nan ,  np.nan,  np.nan,
        np.nan,  np.nan , np.nan,  1.5, np.nan,
        np.nan,  np.nan])

        glm = GLMDemand(
            var_glm_static=["age", "income", "region"],
            var_glm_elasticity=["promo", "loyalty"],
            var_behaviour="price",
            direction="increasing",
            static_penalty_choice=None,
            elasticity_penalty_choice=None,
            nbins=10,
            lam=1e20,
            lam_behaviour=1e5,
            static_offset_betas=offset_static,
            elasticity_offset_betas=None,
        )

        glm.fit(X, y, sample_weight=w)
        summary_glm = glm.summary
        assert np.isclose(summary_glm['static']['coefficients']['region'][1.0], 1.5)
