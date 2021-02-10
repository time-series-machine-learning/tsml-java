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

import java.util.List;
import java.util.Random;

public class RandomSampler implements Sampler{

    private Instances instances;
    private Random random;

    public RandomSampler(Random random){
        this.random = random;
    }

    public RandomSampler(){
        random = new Random();
    }

    public void setInstances(Instances instances) { this.instances = new Instances(instances); }

    public boolean hasNext() { return !instances.isEmpty(); }

    public Instance next() { return instances.remove(random.nextInt(instances.size())); }
}
