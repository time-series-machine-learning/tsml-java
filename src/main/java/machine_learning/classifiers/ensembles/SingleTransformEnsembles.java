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
package machine_learning.classifiers.ensembles;

import tsml.transformers.RowNormalizer;
import tsml.transformers.Transformer;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class SingleTransformEnsembles extends AbstractClassifier{


    enum TransformType {TIME,PS,ACF}; 
    TransformType t = TransformType.TIME;
    Transformer transform;
    Classifier[] classifiers;
    Instances train;

    public SingleTransformEnsembles(){
        super();
        initialise();
    }
    public final void initialise(){
//Transform            
        switch(t){
            case TIME:
                transform=new RowNormalizer();
                break;
                

        }

    }
    @Override
    public void buildClassifier(Instances data){
        
    }        
  
	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}
	public static void main(String[] args){
//Load up Beefand test only on that
		
		
	}
	
}
