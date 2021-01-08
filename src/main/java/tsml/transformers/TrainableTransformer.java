/* 
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.transformers;

import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.Instances;
/**
 * Interface for time series transformers that require training, extending Transformer interface,
 * which does not require a fit stage
 *
 * @author Aaron Bostrom, Tony Bagnall
 * */
public interface TrainableTransformer extends Transformer {

    /**
     * @return true if the the training (fitting) has already happened
     */
    boolean isFit();


    /********* Instances ************/

    /**
     * Build the transform model from train data, storing the necessary info internally.
     * @param data
    */
    void fit(Instances data);

    /**
     *
     * @param data
     * @return
     */
    default Instances fitTransform(Instances data){
        fit(data);
        return transform(data);
    }

    default TimeSeriesInstances fitTransformConverter(Instances data){
        return Converter.fromArff(fitTransform(data));
    }

    /**
     * main transform method. This automatically fits the model if it has not already been fit
     * it effectively implements fitAndTransform. The assumption is that if a Transformer is Trainable,
     * it is not usable until it has been trained/fit
     * @param data
     * @return
     */
    @Override
    default Instances transform(Instances data){
        if(!isFit())
            fit(data);

        return Transformer.super.transform(data);
    }



    /********* TimeSeriesInstances ************/



    /**
     * Build the transform model from train data, storing the necessary info internally.
     * @param data
    */
    void fit(TimeSeriesInstances data);
    
    /**
     *
     * @param data
     * @return
     */
    default TimeSeriesInstances fitTransform(TimeSeriesInstances data){
        fit(data);
        return transform(data);
    }

    default Instances fitTransformConverter(TimeSeriesInstances data){
        return fitTransform(Converter.toArff(data));
    }


            /**
     * main transform method. This automatically fits the model if it has not already been fit
     * it effectively implements fitAndTransform. The assumption is that if a Transformer is Trainable,
     * it is not usable until it has been trained/fit
     * @param data
     * @return
     */
    @Override
    default TimeSeriesInstances transform(TimeSeriesInstances data){
        if(!isFit())
            fit(data);

        return Transformer.super.transform(data);
    }

}