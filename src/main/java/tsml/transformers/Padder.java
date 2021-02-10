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

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class to pad series to make them all equal length. In the ARFF data model, unequal length series are padded with
 * missing values to avoid ragged arrays. This class fills in the series based on the train data. It is assumed the
 * train and test data in ARFF format are already padded to be the same length. This may have involved some prior preprocessing.
 *
 * There is an edge case when the longest series in the Test data is longer than the longest in the Train data. In this
 * scenario, the over long test instance is truncated in Transform.
 *
 * todo: implement univariate
 * todo: implement multivariate
 * todo: handle edge case
 * todo: test all
 * @author Tony Bagnall 18/4/2020
 */
public class Padder implements TrainableTransformer{

    private int finalNumberOfAttributes;
    enum PaddingType{FLAT,NOISE}
    private PaddingType padType=PaddingType.FLAT;

    private boolean isFit;
    /**
     * Finds the length all series will be padded to. Not currently required, but this could be enhanced to remove
     * instances with lengths that could be considered outliers.
     * @param data
     */
    @Override
    public void fit(Instances data) {
        if(data.attribute(0).isRelationValued()) {    //Multivariate
            Instances in=data.instance(0).relationalValue(0);
            finalNumberOfAttributes=in.numAttributes();
        }
        else
            finalNumberOfAttributes=data.numAttributes()-1;

        isFit = true;
    }

    @Override
    public boolean isFit() {
        return isFit;
    }

    /**
     *      * It uses the series mean and variance (if noise is added)
     *      to pad.
     * @param data
     * @return
     */
    @Override
    public Instances transform(Instances data) {
        if(data.attribute(0).isRelationValued()) {    //Multivariate


        }
        else{

        }
        return null;
    }

    @Override
    public Instance transform(Instance inst) {
        return null;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        return null;
    }

    
	@Override
	public TimeSeriesInstance transform(TimeSeriesInstance inst) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void fit(TimeSeriesInstances data) {
		// TODO Auto-generated method stub
		
	}



}
