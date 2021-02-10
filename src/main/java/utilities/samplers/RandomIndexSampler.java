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
 
package utilities.samplers;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RandomIndexSampler implements Sampler{

    private List<Integer> instances;
    private Random random;

    public RandomIndexSampler(Random random){
        this.random = random;
    }

    public RandomIndexSampler(){
        random = new Random();
    }

    public void setInstances(Instances instances) {
        this.instances = new ArrayList(instances.numInstances());
        for (int i = 0; i < instances.numInstances(); i++){
            this.instances.add(i);
        }
    }

    public boolean hasNext() { return !instances.isEmpty(); }

    public Integer next() { return instances.remove(random.nextInt(instances.size())); }
}
