package tsml.transformers;

import java.io.Serializable;

import org.apache.commons.lang3.NotImplementedException;

import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.data_containers.TSCapabilities;
import tsml.data_containers.TSCapabilitiesHandler;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Interface for time series transformers.
 *
 * Previously, we have used weka filters for time series transforms. This redesign
 * is to make our code more like sktime and reduce our Weka dependency. The Filter
 * mechanism has some down sides
 *
 * Ultimately this will include the new data model
 *
 * @author Tony Bagnall 1/1/2020, Aaron Bostrom
 *
 */
public interface Transformer extends TSCapabilitiesHandler, ParamHandler, Serializable {


    /********* Instances ************/

    /**
     * perform the transform process. Some algorithms may require a fit before transform
     * (e.g. shapelets, PCA) others may not (FFT, PAA etc).
     * Should we throw an exception? Default to calling instance transform?
     * Need to determine where to setOut
     * @return Instances of transformed data
     */
    default Instances transform(Instances data){
        Instances output = determineOutputFormat(data);
        for(Instance inst : data){
            output.add(transform(inst));
        }
        return output;
    }

    default TimeSeriesInstances transformConverter(Instances data){
        return transform(Converter.fromArff(data));
    }

    /**
     * Transform a new instance into the format described in determineOutputFormat
     * @param Instance inst
     * @return transformed Instance
     */
    Instance transform(Instance inst);

    /**
     * Method that constructs a holding Instances for transformed data, without doing the transform
     * @param Instances data to derive the format from
     * @return empty Instances to hold transformed Instance objects
     * @throws IllegalArgumentException
     */
    Instances determineOutputFormat(Instances data) throws IllegalArgumentException;

    /**
     * setOptions can be implemented and used in the Tuning mechanism that classifiers employ.
     * It does not have to be implemented though, so there is a default method that throws an exception
     * if it is called
     * @param String array of options that should be in the format of flag,value
     * @throws Exception
     */
    default void setOptions(String[] options) throws Exception{
        throw new NotImplementedException("calling default method of setOptions in Transformer interface, it has not been implemented for class "+this.getClass().getSimpleName());
    }

    /**
     * getCapabilities determines what type of data this Transformer can handle. Default is
     * all numeric  attributes, nominal class, no missing values
     * @return
     */
    default TSCapabilities getTSCapabilities(){
        TSCapabilities result = new TSCapabilities(this);
        result.enable(TSCapabilities.EQUAL_LENGTH)
              .enable(TSCapabilities.MULTI_OR_UNIVARIATE)
              .enable(TSCapabilities.NO_MISSING_VALUES);

        return result;
    }


    /********* TimeSeriesInstances ************/
    
    /**
     * perform the transform process. Some algorithms may require a fit before transform
     * (e.g. shapelets, PCA) others may not (FFT, PAA etc).
     * Should we throw an exception? Default to calling instance transform?
     * Need to determine where to setOut
     * @return Instances of transformed data
     */
    default TimeSeriesInstances transform(TimeSeriesInstances data){
        //when cloning skeleton of TSInstances, copy across classLabels.
        TimeSeriesInstances output = new TimeSeriesInstances(data.getClassLabels());
        for(TimeSeriesInstance inst : data){
            output.add(transform(inst));
        }
        return output;
    }


    default Instances transformConverter(TimeSeriesInstances data){
        return Converter.toArff(transform(data));
    }

    /**
     * Transform a new instance into the format described in determineOutputFormat
     * @param Instance inst
     * @return transformed Instance
     */
    TimeSeriesInstance transform(TimeSeriesInstance inst);
}
