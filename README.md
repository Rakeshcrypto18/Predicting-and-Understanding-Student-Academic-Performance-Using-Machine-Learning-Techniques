# Predicting and Understanding Student Academic Performance Using Machine Learning

This project explores how socioeconomic, behavioral, and academic factors influence student performance in secondary education. Using machine learning techniques, the study identifies key predictors such as attendance, study habits, and parental involvement that significantly impact exam scores.

---

## 📌 Introduction
- Dataset sourced from Kaggle, containing **6,607 records** with **20 features**.  
- Features include attendance, study hours, parental involvement, family income, resource access, extracurricular activities, and more.  
- Objective: Understand factors influencing academic performance and build predictive models to support equitable educational outcomes.

---

## 🎯 Problem Statement
Achievement gaps persist across socioeconomic backgrounds.  
Students from low-income households often face:
- Limited resources  
- Higher absenteeism  
- Reduced parental support  

This project examines how these variables influence student performance and how data-driven insights can help mitigate disparities.

---

## 🔍 Significance
- Analyzes the impact of **family income** on academic success.  
- Evaluates the role of **parental participation**.  
- Investigates **extracurricular activities** and **resource access**.  
- Examines how **teacher actions**, **sleep habits**, and **peer influence** shape outcomes.

---

## 📊 Exploratory Data Analysis (EDA)
Key findings:
- **Attendance (0.58)** and **Study Hours (0.45)** are the strongest positive correlates of exam scores.  
- **Peer Influence:** Positive peer groups significantly boost performance.  
- **Extracurriculars vs Sleep:** Participation introduces variability in sleep patterns.  
- **Internet Access:** Minimal impact overall, but slightly helpful for low- and high-motivation students.

---

## 🤖 Machine Learning Models
1. **Support Vector Regression (SVR):**  
   - Shows positive relationship between resources, extracurriculars, and exam scores.  
   - Limited predictive accuracy but highlights useful trends.  

2. **Random Forest Regressor:**  
   - Used to assess **parental involvement**.  
   - Shows significance of parental support but not sufficient alone to predict performance.  

3. **Gradient Boosting Regressor (with GridSearchCV tuning):**  
   - Achieved **high R² score** and **low MSE**.  
   - Provided the most accurate predictions among tested models.  

---

## ✅ Conclusion
- **Attendance and Study Habits** are the most reliable predictors of academic success.  
- **Socioeconomic factors (family income, parental education)** showed minimal direct impact.  
- **Behavioral and resource-based interventions** are more actionable for improving outcomes.  

---

## 🚀 Future Work
- Incorporate **psychological and environmental factors** (e.g., stress, teacher quality, peer dynamics).  
- Conduct **longitudinal studies** with multimodal data for deeper insights.  
- Implement **structured interventions** like:
  - Attendance monitoring systems  
  - Resource distribution programs  
  - Counseling for stress management  
  - Parental involvement initiatives  

---

## 📚 References
- Chinyoka & Naidu (2013). *Influence of home-based factors on academic performance*  
- Gray, McGuinness & Owende (2014). *Predicting student performance with ML*  
- Jeynes (2007). *Parental involvement and achievement*  
- Kotsiantis et al. (2004). *ML in distance learning*  
- Misopoulos et al. (2017). *Factors affecting performance in online programs*  

---

## 👨‍💻 Author
**Rakesh Somukalahalli Venkatappa**  
Master’s in Data Analytics Engineering, George Mason University  
GitHub: [Rakeshcrypto18](https://github.com/Rakeshcrypto18)  

---
