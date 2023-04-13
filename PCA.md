```python
import numpy as np
import pandas as pd
import random as rd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
```

Hello There!


```python
genes = ['gene' + str(i) for i in range (1, 101)]

wt = ['wt' + str(i) for i in range(1, 6)]
ko = ['ko' + str(i) for i in range(1, 6)]
```


```python
data = pd.DataFrame(columns = [*wt, *ko], index = genes)
for gene in data.index:
    data.loc[gene, 'wt1':'wt5'] = np.random.poisson(lam = rd.randrange(10, 1000), size = 5)
    data.loc[gene, 'ko1':'ko5'] = np.random.poisson(lam = rd.randrange(10, 1000), size = 5)

print(f'Shape of dataset: {data.shape}')
data.head()
```

    Shape of dataset: (100, 10)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wt1</th>
      <th>wt2</th>
      <th>wt3</th>
      <th>wt4</th>
      <th>wt5</th>
      <th>ko1</th>
      <th>ko2</th>
      <th>ko3</th>
      <th>ko4</th>
      <th>ko5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gene1</th>
      <td>425</td>
      <td>450</td>
      <td>461</td>
      <td>428</td>
      <td>421</td>
      <td>68</td>
      <td>53</td>
      <td>44</td>
      <td>54</td>
      <td>50</td>
    </tr>
    <tr>
      <th>gene2</th>
      <td>764</td>
      <td>687</td>
      <td>707</td>
      <td>720</td>
      <td>729</td>
      <td>349</td>
      <td>334</td>
      <td>371</td>
      <td>344</td>
      <td>408</td>
    </tr>
    <tr>
      <th>gene3</th>
      <td>90</td>
      <td>104</td>
      <td>101</td>
      <td>108</td>
      <td>95</td>
      <td>625</td>
      <td>662</td>
      <td>679</td>
      <td>622</td>
      <td>692</td>
    </tr>
    <tr>
      <th>gene4</th>
      <td>227</td>
      <td>250</td>
      <td>293</td>
      <td>263</td>
      <td>245</td>
      <td>527</td>
      <td>528</td>
      <td>579</td>
      <td>523</td>
      <td>570</td>
    </tr>
    <tr>
      <th>gene5</th>
      <td>540</td>
      <td>601</td>
      <td>544</td>
      <td>545</td>
      <td>587</td>
      <td>480</td>
      <td>477</td>
      <td>469</td>
      <td>421</td>
      <td>459</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn import set_config
set_config(transform_output = 'pandas')

ss = StandardScaler()
scaled_data = ss.fit_transform(data.T)
scaled_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gene1</th>
      <th>gene2</th>
      <th>gene3</th>
      <th>gene4</th>
      <th>gene5</th>
      <th>gene6</th>
      <th>gene7</th>
      <th>gene8</th>
      <th>gene9</th>
      <th>gene10</th>
      <th>...</th>
      <th>gene91</th>
      <th>gene92</th>
      <th>gene93</th>
      <th>gene94</th>
      <th>gene95</th>
      <th>gene96</th>
      <th>gene97</th>
      <th>gene98</th>
      <th>gene99</th>
      <th>gene100</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>wt1</th>
      <td>0.935411</td>
      <td>1.223892</td>
      <td>-1.031723</td>
      <td>-1.182566</td>
      <td>0.492526</td>
      <td>0.975024</td>
      <td>1.190554</td>
      <td>-0.889794</td>
      <td>-1.044284</td>
      <td>1.010849</td>
      <td>...</td>
      <td>-0.984260</td>
      <td>0.144167</td>
      <td>1.048750</td>
      <td>-1.272614</td>
      <td>-1.002003</td>
      <td>0.817506</td>
      <td>0.929609</td>
      <td>-0.983812</td>
      <td>-1.074557</td>
      <td>-1.104031</td>
    </tr>
    <tr>
      <th>wt2</th>
      <td>1.065619</td>
      <td>0.800723</td>
      <td>-0.981535</td>
      <td>-1.025799</td>
      <td>1.577151</td>
      <td>1.105740</td>
      <td>0.841993</td>
      <td>-0.991519</td>
      <td>-1.001614</td>
      <td>1.064689</td>
      <td>...</td>
      <td>-1.126107</td>
      <td>-0.576668</td>
      <td>0.931416</td>
      <td>-0.940447</td>
      <td>-0.955317</td>
      <td>0.672172</td>
      <td>0.819846</td>
      <td>-1.005435</td>
      <td>-0.945359</td>
      <td>-0.701100</td>
    </tr>
    <tr>
      <th>wt3</th>
      <td>1.122910</td>
      <td>0.910637</td>
      <td>-0.992290</td>
      <td>-0.732714</td>
      <td>0.563649</td>
      <td>1.010674</td>
      <td>0.972703</td>
      <td>-0.919713</td>
      <td>-0.973167</td>
      <td>0.983928</td>
      <td>...</td>
      <td>-0.996081</td>
      <td>-1.221627</td>
      <td>0.842018</td>
      <td>-1.189573</td>
      <td>-1.019510</td>
      <td>0.890174</td>
      <td>1.140692</td>
      <td>-0.988137</td>
      <td>-0.993347</td>
      <td>-0.660807</td>
    </tr>
    <tr>
      <th>wt4</th>
      <td>0.951036</td>
      <td>0.982081</td>
      <td>-0.967196</td>
      <td>-0.937192</td>
      <td>0.581430</td>
      <td>0.862133</td>
      <td>1.023535</td>
      <td>-0.979551</td>
      <td>-0.995924</td>
      <td>1.078149</td>
      <td>...</td>
      <td>-0.799071</td>
      <td>0.523554</td>
      <td>0.998464</td>
      <td>-0.815885</td>
      <td>-0.978660</td>
      <td>0.599505</td>
      <td>1.030929</td>
      <td>-0.992461</td>
      <td>-0.989655</td>
      <td>-0.963005</td>
    </tr>
    <tr>
      <th>wt5</th>
      <td>0.914578</td>
      <td>1.031543</td>
      <td>-1.013799</td>
      <td>-1.059879</td>
      <td>1.328221</td>
      <td>1.028499</td>
      <td>0.950918</td>
      <td>-1.171033</td>
      <td>-0.970322</td>
      <td>0.853815</td>
      <td>...</td>
      <td>-1.059124</td>
      <td>-0.728423</td>
      <td>1.160498</td>
      <td>-0.670562</td>
      <td>-1.034099</td>
      <td>1.544179</td>
      <td>1.056259</td>
      <td>-1.027057</td>
      <td>-0.985964</td>
      <td>-1.124178</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 100 columns</p>
</div>




```python
pca = PCA()
pca_data = pca.fit_transform(scaled_data)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
labels = ['PC' + str(i) for i in range(1, len(per_var) + 1)]
pca_data.columns = labels
pca_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
      <th>PC9</th>
      <th>PC10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>wt1</th>
      <td>-9.401014</td>
      <td>1.537894</td>
      <td>0.227800</td>
      <td>1.614413</td>
      <td>1.282618</td>
      <td>-0.984772</td>
      <td>-1.275401</td>
      <td>-0.683498</td>
      <td>-0.523661</td>
      <td>7.471672e-16</td>
    </tr>
    <tr>
      <th>wt2</th>
      <td>-9.291936</td>
      <td>-0.396381</td>
      <td>0.275567</td>
      <td>-0.397021</td>
      <td>-1.446900</td>
      <td>-0.576723</td>
      <td>-1.036640</td>
      <td>1.653297</td>
      <td>0.473665</td>
      <td>7.471672e-16</td>
    </tr>
    <tr>
      <th>wt3</th>
      <td>-9.606091</td>
      <td>0.915967</td>
      <td>1.724771</td>
      <td>0.957826</td>
      <td>-1.065200</td>
      <td>1.943675</td>
      <td>0.716454</td>
      <td>-0.406567</td>
      <td>0.175840</td>
      <td>7.471672e-16</td>
    </tr>
    <tr>
      <th>wt4</th>
      <td>-9.602808</td>
      <td>0.350094</td>
      <td>-0.758936</td>
      <td>-0.531413</td>
      <td>0.077588</td>
      <td>-1.467002</td>
      <td>2.063983</td>
      <td>0.010623</td>
      <td>-0.132709</td>
      <td>7.471672e-16</td>
    </tr>
    <tr>
      <th>wt5</th>
      <td>-9.405668</td>
      <td>-2.435085</td>
      <td>-1.495431</td>
      <td>-1.616958</td>
      <td>1.169163</td>
      <td>1.073927</td>
      <td>-0.499680</td>
      <td>-0.559938</td>
      <td>0.038121</td>
      <td>7.471672e-16</td>
    </tr>
    <tr>
      <th>ko1</th>
      <td>9.616821</td>
      <td>-0.463927</td>
      <td>-0.055737</td>
      <td>0.145565</td>
      <td>-0.969942</td>
      <td>-0.833575</td>
      <td>-0.252749</td>
      <td>-1.234347</td>
      <td>1.378658</td>
      <td>7.471672e-16</td>
    </tr>
    <tr>
      <th>ko2</th>
      <td>9.550738</td>
      <td>-2.313374</td>
      <td>-1.196271</td>
      <td>2.500383</td>
      <td>-0.100260</td>
      <td>0.265895</td>
      <td>0.381180</td>
      <td>0.558028</td>
      <td>-0.459838</td>
      <td>7.471672e-16</td>
    </tr>
    <tr>
      <th>ko3</th>
      <td>9.227025</td>
      <td>-1.077022</td>
      <td>2.303394</td>
      <td>-1.419360</td>
      <td>-0.630648</td>
      <td>-0.459906</td>
      <td>-0.159997</td>
      <td>-0.311439</td>
      <td>-1.165360</td>
      <td>7.471672e-16</td>
    </tr>
    <tr>
      <th>ko4</th>
      <td>9.547555</td>
      <td>0.936335</td>
      <td>1.239133</td>
      <td>-0.371433</td>
      <td>2.523760</td>
      <td>0.344199</td>
      <td>0.317527</td>
      <td>0.882917</td>
      <td>0.667995</td>
      <td>7.471672e-16</td>
    </tr>
    <tr>
      <th>ko5</th>
      <td>9.365377</td>
      <td>2.945497</td>
      <td>-2.264289</td>
      <td>-0.882001</td>
      <td>-0.840179</td>
      <td>0.694282</td>
      <td>-0.254676</td>
      <td>0.090924</td>
      <td>-0.452710</td>
      <td>7.471672e-16</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (8,4))
plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.xlabel('Principal component')
plt.ylabel('Percentage of explained variance')
plt.title('Scree Plot')
plt.show()
```


    
![png](output_6_0.png)
    



```python
plt.figure(figsize = (6, 4))
plt.scatter(pca_data.PC1, pca_data.PC2)
plt.xlabel(f'PC1 - {per_var[0]}%')
plt.ylabel(f'PC2 - {per_var[1]}%')
plt.title('PCA Graph')
for sample in pca_data.index:
    plt.annotate(sample , (pca_data.loc[sample, 'PC1'], pca_data.loc[sample, 'PC2']))
plt.show()
```


    
![png](output_7_0.png)
    



```python
loading_scores = pd.Series(pca.components_[0], index = genes)
sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)
top_10_genes = sorted_loading_scores[:10].index.values
loading_scores[top_10_genes]
```




    gene98    0.105623
    gene37   -0.105606
    gene41    0.105604
    gene40    0.105553
    gene73    0.105548
    gene28   -0.105530
    gene14   -0.105527
    gene12    0.105523
    gene69    0.105497
    gene88   -0.105491
    dtype: float64


