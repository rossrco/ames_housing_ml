import tensorflow as tf


INPUT_COLS = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
              'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
              'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
              'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
              'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
              'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
              'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
              'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
              'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
              'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
              '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
              'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
              'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
              'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
              'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
              'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
              'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
              'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
              'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']


OPT_CAT_FEATURES = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond',
                    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
                    'MiscFeature']


OPT_NUM_FEATURES = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']


CAT_FEATURES = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual',
                'ExterCond', 'Foundation', 'Heating', 'HeatingQC',
                'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive',
                'SaleType', 'SaleCondition']


NUM_FEATURES = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
                'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2',
                'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
                'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']


TARGET = 'SalePrice'


RAW_DATA_FEATURE_SPEC = dict(
    [(name, tf.io.FixedLenFeature([], tf.string)) for name in CAT_FEATURES]
    + [(name, tf.io.FixedLenFeature([], tf.float32)) for name in NUM_FEATURES]
    + [(name, tf.io.VarLenFeature(tf.string)) for name in OPT_CAT_FEATURES]
    + [(name, tf.io.VarLenFeature(tf.float32)) for name in OPT_NUM_FEATURES]
    + [(TARGET, tf.io.FixedLenFeature([], tf.float32))])
