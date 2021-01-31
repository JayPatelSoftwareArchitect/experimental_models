import tensorflow as tf
#This class is a wraper for test activation functions, 
class CustomNonlinearActivation(object):
    def __init__(self):
        self.switcher_ = {
            'add_c': self.add_c,
            'add_c2': self.add_c2,
            'add_c3': self.add_c3,
            'sub_c': self.sub_c,
            'sub_c2': self.sub_c2,
            'sub_c3': self.sub_c3,
            'mul_c': self.mul_c,
            'mul_c2': self.mul_c2,
            'mul_c3': self.mul_c3,
            'div_c': self.div_c,
            'div_c2': self.div_c2,
            'div_c3': self.div_c3,
        }
        self.length = self.switcher_.__len__()

    def getActivationFunction(self,activation_type='add_c3'):
        return self.switcher_[activation_type] 

    def add_c(self, value_):
        #log(a) + e^|ib|
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.log(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ + b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_
   
    def add_c2(self, value_):
        #e^a + e^|ib|
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.exp(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ + b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_
   
    def add_c3(self, value_):
        #log(1+a) + e^|ib| 
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.log1p(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ + b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_
   
    def sub_c(self, value_):
        #log(a) * e^|ib|
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.log(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ - b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_
   
    def sub_c2(self, value_):
        #e^a - e^|ib|
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.exp(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ - b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_
   
    def sub_c3(self, value_):
        #log(1+a) - e^|ib| 
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.log1p(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ - b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_

    def mul_c(self, value_):
        #log(a) * e^|ib|
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.log(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ * b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_
   
    def mul_c2(self, value_):
        #e^a * e^|ib|
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.exp(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ * b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_
   
    def mul_c3(self, value_):
        #log(1+a) * e^|ib| 
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.log1p(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ * b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_
   
    def div_c(self, value_):
        #log(a) / e^|ib|
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.log(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ / b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_
   
    def div_c2(self, value_):
        #e^a / e^|ib|
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.exp(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ / b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_
    
    def div_c3(self, value_):
        #log(1+a) / e^|ib| 
        value_ = tf.dtypes.cast(value_, tf.complex64)
        a_ = tf.math.log1p(tf.math.real(value_))
        
        b_ = tf.math.exp(tf.math.imag(value_))
        
        a_ = tf.where(tf.math.is_nan(a_), tf.ones_like(a_) , a_)
        b_ = tf.where(tf.math.is_nan(b_), tf.ones_like(b_) , b_)
        r_ = a_ / b_
        r_ = tf.where(tf.math.is_nan(r_), tf.ones_like(r_), r_)
        return r_


