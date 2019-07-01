package timeseriesweka.elastic_distance_measures;

/*
 * Move Split Merge distance measure from
@ARTICLE{stefan13move-split-merge,     AUTHOR = "A. Stefan andf V. Athitsos and G. Das ",
        TITLE = "The Move-Split-Merge Metric for Time Series",
        JOURNAL = "{IEEE} TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING",
        YEAR = "2013",
        VOLUME = "25 ",
        NUMBER = "6 ",
        PAGES="1425--1438" }

*
 */

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author Chris Rimmer
 */
public class MSMDistance extends EuclideanDistance{

    // c - cost of Split/Merge operation. Change this value to what is more
		// appropriate for your data.
    double c = 0.1;
    public MSMDistance(){
        super();
        this.m_DontNormalize = true;
    }
    public MSMDistance(double c){
        super();
        this.m_DontNormalize = true;
        this.c = c;
    }
    public void setC(double v){c=v;}

    /**
     * Distance method
     *
     * @param first instance 1
     * @param second instance 2
     * @param cutOffValue used for early abandon
     * @param stats
     * @return distance between instances
     */
    @Override
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats){
        //Get the double arrays
        return distance(first,second,cutOffValue);
    }

    /**
     * distance method that converts instances to arrays of doubles
     *
     * @param first instance 1
     * @param second instance 2
     * @param cutOffValue used for early abandon
     * @return distance between instances
     */
    @Override
    public double distance(Instance first, Instance second, double cutOffValue){
        //remove class index from first instance if there is one
        int firtClassIndex = first.classIndex();
        double[] arr1;
        if(firtClassIndex > 0){
            arr1 = new double[first.numAttributes()-1];
            for(int i = 0,j = 0; i < first.numAttributes(); i++){
                if(i != firtClassIndex){
                    arr1[j]= first.value(i);
                    j++;
                }
            }
        }else{
            arr1 = first.toDoubleArray();
        }

        //remove class index from second instance if there is one
        int secondClassIndex = second.classIndex();
        double[] arr2;
        if(secondClassIndex > 0){
            arr2 = new double[second.numAttributes()-1];
            for(int i = 0,j = 0; i < second.numAttributes(); i++){
                if(i != secondClassIndex){
                    arr2[j]= second.value(i);
                    j++;
                }
            }
        }else{
            arr2 = second.toDoubleArray();
        }

        return distance(arr1,arr2,cutOffValue);
    }

    /**
     * calculates the distance between two instances (been converted to arrays)
     * Exact code from the authors downloaded from
     * http://omega.uta.edu/~athitsos/msm/
     *
     * @param first instance 1 as array
     * @param second instance 2 as array
     * @param cutOffValue used for early abandon
     * @return distance between instances
     */
    public double MSM_Distance(double[] a, double[] b){

        int m, n, i, j;
        m = a.length;
        n = b.length;

        double[][] cost = new double[m][n];

        // Initialization
        cost[0][0] = Math.abs(a[0] - b[0]);

        for (i = 1; i< m; i++) {
            cost[i][0] = cost[i-1][0] + editCost(a[i], a[i-1], b[0]);
        }

        for (j = 1; j < n; j++) {
            cost[0][j] = cost[0][j-1] + editCost(b[j], a[0], b[j-1]);
        }

        // Main Loop
        for( i = 1; i < m; i++){
            for ( j = 1; j < n; j++){
                double d1,d2, d3;
                d1 = cost[i-1][j-1] + Math.abs(a[i] - b[j] );
                d2 = cost[i-1][j] + editCost(a[i], a[i-1], b[j]);
                d3 = cost[i][j-1] + editCost(b[j], a[i], b[j-1]);
                cost[i][j] = Math.min( d1, Math.min(d2,d3) );
            }
        }

        // Output
        return cost[m-1][n-1];
    }


    public double editCost( double new_point, double x, double y){
        double dist = 0;

        if ( ( (x <= new_point) && (new_point <= y) ) ||
             ( (y <= new_point) && (new_point <= x) ) ) {
            dist = c;
        }
        else{
                dist = c + Math.min( Math.abs(new_point - x) , Math.abs(new_point - y) );
        }

        return dist;
    }



    public double distance(double[] first, double[] second, double cutOffValue){
        return MSM_Distance(first,second);
    }


}