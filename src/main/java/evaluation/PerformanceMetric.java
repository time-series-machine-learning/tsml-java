/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package evaluation;

import evaluation.storage.ClassifierResults;
import java.util.ArrayList;
import java.util.function.Function;

/**
 * This is essentially a placeholder class, and may be expanded into a full package
 * with this being an abstract base class etc. 
 * 
 * For now, this is a container class for metrics and meta info about them. It contains
 *     - The name of this metric for printouts
 *     - A function to get this metric's score fro ma classifier results object (in future, perhaps calculate them here instead, etc)
 *     - A flag for whether this metric wants to be maximised or minimised
 *     - A flag to _suggest_ how this metric should be summarised/averaged 
 *                - for now, mean vs median for e.g accs vs timings. For timings we would want to 
 *                  use median instead of mean to reduce effect of outliers
 *                - in future, probably just define a comparator
 *     - Maybe more in the future     
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class PerformanceMetric {

    public String metricName;
    public Function<ClassifierResults, Double> getter;
    public boolean takeMean;
    public boolean maximised;
    
    public PerformanceMetric(String metricName, Function<ClassifierResults, Double> getScore, boolean takeMean, boolean maximised) {
        this.metricName = metricName;
        this.getter = getScore;
        this.takeMean = takeMean;
        this.maximised = maximised;
    }
    
    public double getScore(ClassifierResults res) {
        return getter.apply(res);
    }
    
    public static PerformanceMetric acc             = new PerformanceMetric("ACC", ClassifierResults.GETTER_Accuracy, true, true);
    public static PerformanceMetric balacc          = new PerformanceMetric("BALACC", ClassifierResults.GETTER_BalancedAccuracy, true, true);
    public static PerformanceMetric AUROC           = new PerformanceMetric("AUROC", ClassifierResults.GETTER_AUROC, true, true);
    public static PerformanceMetric NLL             = new PerformanceMetric("NLL", ClassifierResults.GETTER_NLL, true, true);
    public static PerformanceMetric F1              = new PerformanceMetric("F1", ClassifierResults.GETTER_F1, true, true);
    public static PerformanceMetric MCC             = new PerformanceMetric("MCC", ClassifierResults.GETTER_MCC, true, true);
    public static PerformanceMetric precision       = new PerformanceMetric("Prec", ClassifierResults.GETTER_Precision, true, true);
    public static PerformanceMetric recall          = new PerformanceMetric("Recall", ClassifierResults.GETTER_Recall, true, true);
    public static PerformanceMetric sensitivity     = new PerformanceMetric("Sens", ClassifierResults.GETTER_Sensitivity, true, true);
    public static PerformanceMetric specificity     = new PerformanceMetric("Spec", ClassifierResults.GETTER_Specificity, true, true);
    public static PerformanceMetric buildTime       = new PerformanceMetric("TrainTime", ClassifierResults.GETTER_buildTimeDoubleMillis, true, true);
    public static PerformanceMetric testTime        = new PerformanceMetric("TestTime", ClassifierResults.GETTER_testTimeDoubleMillis, true, true);
    
    
    public static ArrayList<PerformanceMetric> getDefaultStatistics() { 
        ArrayList<PerformanceMetric> stats = new ArrayList<>();
        stats.add(acc);
        stats.add(balacc);
        stats.add(AUROC);
        stats.add(NLL);
        return stats;
    }
        
    public static ArrayList<PerformanceMetric> getAllStatistics() { 
        ArrayList<PerformanceMetric> stats = new ArrayList<>();
        stats.add(acc);
        stats.add(balacc);
        stats.add(AUROC);
        stats.add(NLL);
        stats.add(F1);
        stats.add(MCC);
        stats.add(precision);
        stats.add(recall);
        stats.add(sensitivity);
        stats.add(specificity);
        return stats;
    }
}
