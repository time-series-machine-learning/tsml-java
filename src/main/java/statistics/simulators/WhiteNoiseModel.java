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
 

/*
Basic model that just adds normal noise. This is actually the default behaviour in 
model, so we can abstract model and do it here instead (at a later date)
 */
package statistics.simulators;

/**
 *
 * @author ajb
 */
public class WhiteNoiseModel extends  Model{

    public WhiteNoiseModel(){
        super();
    }

    @Override
    public void setParameters(double[] p) {//Mean and variance of the noise
        setVariance(p[0]);
        
    }
    
    
}
