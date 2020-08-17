/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package experiments;

import tsml.transformers.*;
import tsml.filters.shapelet_filters.ShapeletFilter;

/**
 *
 * @author Aaron Bostrom and Tony Bagnall
 */
public class TransformLists {

    //All implemented time series related SimpleBatchFilters in tsml
    //<editor-fold defaultstate="collapsed" desc="All time series related SimpleBatchFilters">
    public static String[] allFilters={
            "ACF","ACF_PACF","ARMA","BagOfPatterns","BinaryTransform","Clipping",
            "Cosine","Derivative","Differences","Fast_FFT", "FFT","Hilbert","MatrixProfile",
            "NormalizeCase","PAA","PACF","PowerCepstrum","RankOrder",
            "RunLength","SAX","Sine","SummaryStats","ShapeletTransform"
    };
    //</editor-fold>
    //multivariate SimpleBatchFilters in tsml
    //<editor-fold defaultstate="collapsed" desc="Filters that transform univariate into multivariate">
    public static String[] multivariateFilters={"Spectrogram","MFCC"};
    //</editor-fold>

    public static Transformer setTransform(Experiments.ExperimentalArguments exp){
        return setClassicTransform(exp.classifierName, exp.foldId);
    }


    //TODO: Fix for new Transformers.
    public static Transformer setClassicTransform(String transformName, int foldId) {
        Transformer transformer = null;
        switch(transformName){
            case "ShapeletTransform": case "ST":
                transformer = new ShapeletTransform();
                break;
            case "ACF":
               transformer = new ACF();
               break;
            case "ACF_PACF":
               transformer = new ACF_PACF();
               break;
            case "ARMA":
               transformer = new ARMA();
               break;
            case "BagOfPatterns":
               transformer = new BagOfPatterns();
               break;
            case "BinaryTransform":
               transformer = new BinaryTransform();
               break;
            case "Clipping":
               transformer = new Clipping();
               break;
            case "Cosine":
                transformer = new Cosine();
                break;
            case "Derivative":
               transformer = new Derivative();
               break;
            case "Differences":
               transformer = new Differences();
               break;
            case "Fast_FFT":
               transformer = new Fast_FFT();
               break;
            case "FFT":
               transformer = new FFT();
               break;
            case "Hilbert":
                transformer = new Hilbert();
                break;
            case "MatrixProfile":
               transformer = new MatrixProfile();
               break;
            case "MFCC":
               transformer = new MFCC();
               break;
            case "NormalizeCase":
                transformer = new RowNormalizer();
                break;
            case "PAA":
               transformer = new PAA();
               break;
            case "PACF":
               transformer = new PACF();
               break;
            case "PowerCepstrum":
               transformer = new PowerCepstrum();
               break;
            case "PowerSpectrum":
               transformer = new PowerSpectrum();
               break;
            case "RankOrder":
               transformer = new RankOrder();
               break;
            case "RunLength":
               transformer = new RunLength();
               break;
            case "SAX":
               transformer = new SAX();
               break;
            case "Sine":
                transformer = new Sine();
                break;
            case "Spectrogram":
               transformer = new Spectrogram();
               break;
            case "SummaryStats":
               transformer = new SummaryStats();
               break;



            default:
                System.out.println("UNKNOWN TRANSFORM "+transformName);
                System.exit(0);
        }
        
        return transformer;
    }
    
   public static void main(String[] args) throws Exception {
        System.out.println(setClassicTransform("ST", 0));
        System.out.println(setClassicTransform("ShapeletTransform", 0));
    }
    
    
    
}
