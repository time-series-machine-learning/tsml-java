package tsml.classifiers.shapelet_based.classifiers;

import tsml.data_containers.TimeSeries;

public class ShapeletDataMV {

    double[][] data;

    int[] seriesIndex;
    int n,m;
    double quality;

    public ShapeletDataMV(double[][] data, int[] seriesIndex, int n, int m ){
        this.data = data;
        this.seriesIndex = seriesIndex;
    }
    public ShapeletDataMV(int n, int m ){
       this.n = n;
       this.m = m;
       this.data = new double[n][m];
       this.seriesIndex = new int[m];
    }

    public void setData(double value, int i, int j){
        this.data[i][j] = value;
    }

    public void setSeriesIndex(int value, int j){
        this.seriesIndex[j] = value;
    }

    public void setQuality(double q){
        this.quality = q;
    }

    public double distanceByIndex(TimeSeries series, int index){
        double dist = Double.MAX_VALUE;

        for (int i=0;i<series.getSeriesLength()-n;i++){
            double d = 0;
            for (int j=0;j<n;j++){
                d+= (series.get(i+j)*this.data[j][index])*(series.get(i+j)*this.data[j][index]);
            }
            if (d<dist){
                dist = d;
            }
        }
        return Math.sqrt(dist);
    }


}
