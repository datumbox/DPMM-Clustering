/* 
 * Copyright (C) 2014 Vasilis Vryniotis <bbriniotis at datumbox.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package com.datumbox.opensource.dataobjects;

import org.apache.commons.math3.linear.RealVector;

/**
 * Point Object is used to store the id and the data of the xi record.
 * 
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 */
public class Point {
    /**
     * The id variable is used to identify the xi record.
     */
    public Integer id;
    
    /**
     * The data variable is a RealVector which stores the information of xi record.
     */
    public RealVector data;
    
    /**
     * Point Constructor which accepts a RealVector input.
     * 
     * @param id    The integer id of the point
     * @param data  The data of the point
     */
    public Point(Integer id, RealVector data)  {
        this.id = id;
        this.data = data;
    }
}
