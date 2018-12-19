/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka.measures;

import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author Aaron
 */
public class EuclideanDistance_I extends EuclideanDistance{
       
    public EuclideanDistance_I(){}
    
    public EuclideanDistance_I(Instances train){
        super(train);
        
        m_Data = null;
        m_Validated = true;
    }
    
    

    @Override
    public double distance(Instance multiSeries1, Instance multiseries2, double cutoff){
        
        //split the instance.
        Instance[] multi1 = splitMultivariateInstance(multiSeries1);
        Instance[] multi2 = splitMultivariateInstance(multiseries2);

        //TODO: might need to normalise here.
        double[][] data1 = utilities.multivariate_tools.MultivariateInstanceTools.convertMultiInstanceToTransposedArrays(multi1);
        double[][] data2 = utilities.multivariate_tools.MultivariateInstanceTools.convertMultiInstanceToTransposedArrays(multi2);
        return distance(data1, data2, cutoff);
    }
    
    public double distance(double[][] a, double[][] b, double cutoff){
        //assume a and b are the same length.
        double sum =0;
        for(int i=0; i<a.length; i++){
            sum += Math.sqrt(sqMultiDist(a[i],b[i]));
        }
        return sum;
    }
    
    double sqDist(double a, double b){
        return (a-b)*(a-b);
    }
    
    //given each aligned value in the channel.
    double sqMultiDist(double[] a, double[] b){
        double sum = 0;
        for(int i=0; i<a.length; i++){
            sum += sqDist(a[i], b[i]);
        }
        return sum;
    }
}
