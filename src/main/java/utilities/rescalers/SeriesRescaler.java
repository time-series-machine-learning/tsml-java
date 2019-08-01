/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.rescalers;

/**
 *
 * @author a.bostrom1
 */
public interface SeriesRescaler {
   
    
    public double[] rescaleSeries(double[] series);
    public double[] rescaleSeries(double[] series, boolean hasClassValue);
}
