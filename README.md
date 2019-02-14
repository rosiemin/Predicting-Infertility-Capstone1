**********************************************
# Predicting Infertility from Self-Reported Data
**********************************************

#### Author: Rosie M Martinez, ScD, MPH
#### Website: https://github.com/rosiemin/dsi-capstone1

# Background
Human infertility is a complex disorder that is becoming more prevalent. In 2010, an estimated 48.5 million couples worldwide were unable to have a child after five years of trying to conceive. Worldwide 1 in 6 couples have trouble getting pregnant or sustaining their pregnancy and approximately 6.9 million American women struggle with issues of infertility. According to the National Center for Health Statistics, in the United States 12.1% of women aged 15-44 have impaired fecundity (the ability to have kids) and 6.7% of married women in the same age range are infertile.

<p align="center">
<img src="images/fertility.jpg">

[ source](https://www.stanfordchildrens.org/content-public/pdf/national-infertility-week-stanford-childrens.pdf)
</p>


Infertility is influenced by a broad range of physical, anatomical, hormonal, genetic and environmental stressors. About 1/3 of infertility is attributed to female issues, 1/3 is attributed to male issues, and 1/3 is attributed to unknown factors. The burden of infertility high and remains an ongoing global reproductive health issue.

The clinical diagnosis of infertility is defined as the failure to conceive within 12 months and affects 7% to 8% of reproductive-aged American women. The American Society for Reproductive Medicine recommends that a woman should consult her physician if she is under 35 years of age and has been trying to conceive for more than 12 months or over 35 years of age and has been trying for 6 or more months.

### Question: Can I predict infertility among women based on self-reported risk factor data?

# Data
![](images/nhanes_logo.png)
[ source](https://www.cdc.gov/nchs/nhanes/)

The National Health and Nutrition Examination Survey, or NHANES as it is better known, is a program designed to assess the health and nutritional status of adults and children in the United States. The survey examines a nationally representative sample of about 5,000 persons each year. These persons are located in counties across the country, 15 of which are visited each year. The NHANES interview includes demographic, socioeconomic, dietary, and health-related questions. The examination component consists of medical, dental, and physiological measurements, as well as laboratory tests administered by highly trained medical personnel.

While we infertility is not only a “female” issue, for the purposes of this capstone, I will be focusing on infertility among women. In this data, there are many different datasets based on questionnaires given, but I will be focusing my project on those questions dealing with the known risk factors of infertility, including
<p align="center">
<img src="images/riskfactors.png" width=500px>
</p>

* Age
* Race/Ethnicity
* BMI (body mass index)
*	Pre-existing hormonal issues, STIs, and pelvic inflammatory disease
*	Physical activity
*	Alcohol consumption
*	Smoking
*	Irregular periods

For the purposes of this analysis I have excluded women who have had a hysterectomy, or have had uterine, cervical, or ovarian cancer.

# Analysis

#### How did I define infertility?
In the NHANES dataset, there are two questions asked:
* “Have you/spouse ever attempted to become pregnant over a period of at least a year without becoming pregnant?”
* “Have you or your spouse ever been to a doctor or other medical provider because you or she has been unable to become pregnant?”

Those participants that said “Yes” to one of either of the two questions are infertile and those that said “No” as fertile.

### EDA:
*note, I'm only showing EDA on those variables I kept in my final model*

#### Continuous Variables: Age, BMI, Alcohol
<p align="center">
<img src="images/pairplot.png" width=600px>
<img src="images/boxplot_by.png" width=600px>
</p>


#### Categorical Variables: STIs (yes/no), Irregular Periods (yes/no), Smoking Status (Current/Former/Never)
<p align="center">
<img src="images/proportionalcats.png" width=600px>
</p>

### Model Training:
<p align="center">
<img src="images/flowchart.png" width = 600px>
</p>

#### How did I choose my test metric?
1. What is the question at hand?
2. What are the potential metrics?

### Final Model Assessment:

```python
Coefficients from the final holdout model using Stats Models
                      coef     P>|z|
Intercept          -5.0404     0.000      
Age                 0.0394     0.047      
Alcohol             0.0087     0.881     
BMI                 0.0513     0.035      
STI_yes            -0.3440     0.599     
Irr Periods_yes    -0.1853     0.737      
Current Smoke      -0.0322     0.949    
Former Smoke       -0.7065     0.289     
```
Because age and BMI are significant, we can look at the exp(0.0394) = 1.04 (age), and exp(0.0513) = 1.05

***Interpretation of age***
**Holding all else constant, on average, with a one year increase in age, the odds increase by 4% among those that are infertile compared to those that are fertile.**

***Interpretation of BMI***

**Holding all else constant, on average, with a one unit increase in BMI, the odds increase by 5% among those that are infertile compared to those that are fertile.**

<p align="center">
<img src="images/ROC_holdout.png">
</p>

* **The recall from my training model 0.906**
* **The recall from my final hold out was 0.886**

<p align="center">
<img src="images/confusion_mat_holdout.png">
</p>


# Discussion



# Future Steps
* Re-do analysis including the other years that NHANES has available (approximately 10 more years worth of data).
*
