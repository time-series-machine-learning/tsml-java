/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.rescalers;

/**
 *
 * @author a.bostrom1
 * 
 * 
 * This class just wraps up the series rescaler for no rescaling. 
 * It allows the user to obfuscate to using classes what type of rescaling we're doing
 * as they shouldn't care.
 */
public class NoRescaling implements SeriesRescaler{

    @Override
    public double[] rescaleSeries(double[] series) {
        return rescaleSeries(series, false);
    }

    @Override
    public double[] rescaleSeries(double[] series, boolean hasClassValue) {
        return series;
    }
    
}
