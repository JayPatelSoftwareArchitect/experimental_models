йЄ
й
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
S
Imag

input"T
output"Tout"
Ttype0:
2"
Touttype0:
2
+
IsNan
x"T
y
"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
S
Real

input"T
output"Tout"
Ttype0:
2"
Touttype0:
2
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8оо

x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	А
*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0

Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_10/kernel/m

*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:*
dtype0

Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*'
shared_nameAdam/dense_11/kernel/m

*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes
:	А
*
dtype0

Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0

Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_10/kernel/v

*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:*
dtype0

Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*'
shared_nameAdam/dense_11/kernel/v

*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes
:	А
*
dtype0

Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
ц)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ё)
value)B) B)

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
Ќ
&iter

'beta_1

(beta_2
	)decay
*learning_ratemTmUmVmW mX!mYvZv[v\v] v^!v_
 
*
0
1
2
3
 4
!5
*
0
1
2
3
 4
!5
­
regularization_losses
+layer_regularization_losses
,non_trainable_variables
	variables

-layers
	trainable_variables
.layer_metrics
/metrics
 
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
0layer_regularization_losses
1non_trainable_variables
	variables

2layers
trainable_variables
3layer_metrics
4metrics
 
 
 
­
regularization_losses
5layer_regularization_losses
6non_trainable_variables
	variables

7layers
trainable_variables
8layer_metrics
9metrics
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
:layer_regularization_losses
;non_trainable_variables
	variables

<layers
trainable_variables
=layer_metrics
>metrics
 
 
 
­
regularization_losses
?layer_regularization_losses
@non_trainable_variables
	variables

Alayers
trainable_variables
Blayer_metrics
Cmetrics
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
­
"regularization_losses
Dlayer_regularization_losses
Enon_trainable_variables
#	variables

Flayers
$trainable_variables
Glayer_metrics
Hmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
#
0
1
2
3
4
 

I0
J1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ktotal
	Lcount
M	variables
N	keras_api
D
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

M	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

O0
P1

R	variables
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_dense_9_inputPlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
Ђ
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_9_inputdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_162287
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ѕ

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_163083

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/mAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_163174хэ	
=

__inference__traced_save_163083
file_prefix-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*О
_input_shapesЌ
Љ: :::::	А
:
: : : : : : : : : :::::	А
:
:::::	А
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	А
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	А
: 

_output_shapes
:
:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	А
: 

_output_shapes
:
:

_output_shapes
: 
Д1
т
C__inference_dense_9_layer_call_and_return_conditional_losses_162852

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
BiasAddk
CastCastBiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
CastK
RealRealCast:y:0*+
_output_shapes
:џџџџџџџџџ2
RealV
ExpExpReal:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ExpK
ImagImagCast:y:0*+
_output_shapes
:џџџџџџџџџ2
ImagZ
Exp_1ExpImag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Exp_1V
IsNanIsNanExp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
IsNanY
ones_like/ShapeShapeExp:y:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	ones_like~
SelectV2SelectV2	IsNan:y:0ones_like:output:0Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2\
IsNan_1IsNan	Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
IsNan_1_
ones_like_1/ShapeShape	Exp_1:y:0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ones_like_1

SelectV2_1SelectV2IsNan_1:y:0ones_like_1:output:0	Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2_1o
mulMulSelectV2:output:0SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
mulZ
IsNan_2IsNanmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
IsNan_2]
ones_like_2/ShapeShapemul:z:0*
T0*
_output_shapes
:2
ones_like_2/Shapek
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_2/Const
ones_like_2Fillones_like_2/Shape:output:0ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ones_like_2

SelectV2_2SelectV2IsNan_2:y:0ones_like_2:output:0mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2_2
IdentityIdentitySelectV2_2:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
в
М
$__inference_signature_wrapper_162287
dense_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_1619522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namedense_9_input

Щ
H__inference_sequential_3_layer_call_and_return_conditional_losses_162402

inputs-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*dense_10_tensordot_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identityЂdense_10/BiasAdd/ReadVariableOpЂ!dense_10/Tensordot/ReadVariableOpЂdense_11/BiasAdd/ReadVariableOpЂdense_11/MatMul/ReadVariableOpЂdense_9/BiasAdd/ReadVariableOpЂ dense_9/Tensordot/ReadVariableOpЎ
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axes
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/freeh
dense_9/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_9/Tensordot/Shape
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisљ
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axisџ
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1Ј
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisи
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatЌ
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stackЈ
dense_9/Tensordot/transpose	Transposeinputs!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Tensordot/transposeП
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_9/Tensordot/ReshapeО
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_9/Tensordot/MatMul
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/Const_2
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisх
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1А
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/TensordotЄ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpЇ
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/BiasAdd
dense_9/CastCastdense_9/BiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Castc
dense_9/RealRealdense_9/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Realn
dense_9/ExpExpdense_9/Real:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Expc
dense_9/ImagImagdense_9/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Imagr
dense_9/Exp_1Expdense_9/Imag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Exp_1n
dense_9/IsNanIsNandense_9/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNanq
dense_9/ones_like/ShapeShapedense_9/Exp:y:0*
T0*
_output_shapes
:2
dense_9/ones_like/Shapew
dense_9/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like/ConstЈ
dense_9/ones_likeFill dense_9/ones_like/Shape:output:0 dense_9/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_likeІ
dense_9/SelectV2SelectV2dense_9/IsNan:y:0dense_9/ones_like:output:0dense_9/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2t
dense_9/IsNan_1IsNandense_9/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNan_1w
dense_9/ones_like_1/ShapeShapedense_9/Exp_1:y:0*
T0*
_output_shapes
:2
dense_9/ones_like_1/Shape{
dense_9/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like_1/ConstА
dense_9/ones_like_1Fill"dense_9/ones_like_1/Shape:output:0"dense_9/ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_like_1А
dense_9/SelectV2_1SelectV2dense_9/IsNan_1:y:0dense_9/ones_like_1:output:0dense_9/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2_1
dense_9/mulMuldense_9/SelectV2:output:0dense_9/SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/mulr
dense_9/IsNan_2IsNandense_9/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNan_2u
dense_9/ones_like_2/ShapeShapedense_9/mul:z:0*
T0*
_output_shapes
:2
dense_9/ones_like_2/Shape{
dense_9/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like_2/ConstА
dense_9/ones_like_2Fill"dense_9/ones_like_2/Shape:output:0"dense_9/ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_like_2Ў
dense_9/SelectV2_2SelectV2dense_9/IsNan_2:y:0dense_9/ones_like_2:output:0dense_9/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2_2w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout_3/dropout/ConstЊ
dropout_3/dropout/MulMuldense_9/SelectV2_2:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/Mul}
dropout_3/dropout/ShapeShapedense_9/SelectV2_2:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeж
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dropout_3/dropout/GreaterEqual/yъ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2 
dropout_3/dropout/GreaterEqualЁ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/CastІ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/Mul_1Б
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/free
dense_10/Tensordot/ShapeShapedropout_3/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axisў
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2_1~
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/ConstЄ
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1Ќ
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axisн
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concatА
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stackР
dense_10/Tensordot/transpose	Transposedropout_3/dropout/Mul_1:z:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Tensordot/transposeУ
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_10/Tensordot/ReshapeТ
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_10/Tensordot/MatMul
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/Const_2
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisъ
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1Д
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/TensordotЇ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpЋ
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/BiasAdd
dense_10/CastCastdense_10/BiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Castf
dense_10/RealRealdense_10/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Realq
dense_10/ExpExpdense_10/Real:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Expf
dense_10/ImagImagdense_10/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Imagu
dense_10/Exp_1Expdense_10/Imag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Exp_1q
dense_10/IsNanIsNandense_10/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNant
dense_10/ones_like/ShapeShapedense_10/Exp:y:0*
T0*
_output_shapes
:2
dense_10/ones_like/Shapey
dense_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like/ConstЌ
dense_10/ones_likeFill!dense_10/ones_like/Shape:output:0!dense_10/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_likeЋ
dense_10/SelectV2SelectV2dense_10/IsNan:y:0dense_10/ones_like:output:0dense_10/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2w
dense_10/IsNan_1IsNandense_10/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNan_1z
dense_10/ones_like_1/ShapeShapedense_10/Exp_1:y:0*
T0*
_output_shapes
:2
dense_10/ones_like_1/Shape}
dense_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like_1/ConstД
dense_10/ones_like_1Fill#dense_10/ones_like_1/Shape:output:0#dense_10/ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_like_1Е
dense_10/SelectV2_1SelectV2dense_10/IsNan_1:y:0dense_10/ones_like_1:output:0dense_10/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2_1
dense_10/mulMuldense_10/SelectV2:output:0dense_10/SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/mulu
dense_10/IsNan_2IsNandense_10/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNan_2x
dense_10/ones_like_2/ShapeShapedense_10/mul:z:0*
T0*
_output_shapes
:2
dense_10/ones_like_2/Shape}
dense_10/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like_2/ConstД
dense_10/ones_like_2Fill#dense_10/ones_like_2/Shape:output:0#dense_10/ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_like_2Г
dense_10/SelectV2_2SelectV2dense_10/IsNan_2:y:0dense_10/ones_like_2:output:0dense_10/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2_2s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ0  2
flatten_3/Const
flatten_3/ReshapeReshapedense_10/SelectV2_2:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџА2
flatten_3/ReshapeЉ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02 
dense_11/MatMul/ReadVariableOpЂ
dense_11/MatMulMatMulflatten_3/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/MatMulЇ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_11/BiasAdd/ReadVariableOpЅ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/SoftmaxЛ
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е1
у
D__inference_dense_10_layer_call_and_return_conditional_losses_162939

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
BiasAddk
CastCastBiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
CastK
RealRealCast:y:0*+
_output_shapes
:џџџџџџџџџ2
RealV
ExpExpReal:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ExpK
ImagImagCast:y:0*+
_output_shapes
:џџџџџџџџџ2
ImagZ
Exp_1ExpImag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Exp_1V
IsNanIsNanExp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
IsNanY
ones_like/ShapeShapeExp:y:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	ones_like~
SelectV2SelectV2	IsNan:y:0ones_like:output:0Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2\
IsNan_1IsNan	Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
IsNan_1_
ones_like_1/ShapeShape	Exp_1:y:0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ones_like_1

SelectV2_1SelectV2IsNan_1:y:0ones_like_1:output:0	Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2_1o
mulMulSelectV2:output:0SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
mulZ
IsNan_2IsNanmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
IsNan_2]
ones_like_2/ShapeShapemul:z:0*
T0*
_output_shapes
:2
ones_like_2/Shapek
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_2/Const
ones_like_2Fillones_like_2/Shape:output:0ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ones_like_2

SelectV2_2SelectV2IsNan_2:y:0ones_like_2:output:0mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2_2
IdentityIdentitySelectV2_2:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
Ы
H__inference_sequential_3_layer_call_and_return_conditional_losses_162245

inputs
dense_9_162227
dense_9_162229
dense_10_162233
dense_10_162235
dense_11_162239
dense_11_162241
identityЂ dense_10/StatefulPartitionedCallЂ dense_11/StatefulPartitionedCallЂdense_9/StatefulPartitionedCall
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_162227dense_9_162229*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_1620072!
dense_9/StatefulPartitionedCall§
dropout_3/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1620402
dropout_3/PartitionedCallД
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_10_162233dense_10_162235*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_1621042"
 dense_10/StatefulPartitionedCallћ
flatten_3/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1621262
flatten_3/PartitionedCallА
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_11_162239dense_11_162241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_1621452"
 dense_11/StatefulPartitionedCallх
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А
c
*__inference_dropout_3_layer_call_fn_162883

inputs
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1620352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

F
*__inference_flatten_3_layer_call_fn_162959

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1621262
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџА2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ	
н
D__inference_dense_11_layer_call_and_return_conditional_losses_162145

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџА::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџА
 
_user_specified_nameinputs
ђ
а
H__inference_sequential_3_layer_call_and_return_conditional_losses_162767
dense_9_input-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*dense_10_tensordot_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identityЂdense_10/BiasAdd/ReadVariableOpЂ!dense_10/Tensordot/ReadVariableOpЂdense_11/BiasAdd/ReadVariableOpЂdense_11/MatMul/ReadVariableOpЂdense_9/BiasAdd/ReadVariableOpЂ dense_9/Tensordot/ReadVariableOpЎ
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axes
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/freeo
dense_9/Tensordot/ShapeShapedense_9_input*
T0*
_output_shapes
:2
dense_9/Tensordot/Shape
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisљ
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axisџ
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1Ј
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisи
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatЌ
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stackЏ
dense_9/Tensordot/transpose	Transposedense_9_input!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Tensordot/transposeП
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_9/Tensordot/ReshapeО
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_9/Tensordot/MatMul
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/Const_2
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisх
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1А
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/TensordotЄ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpЇ
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/BiasAdd
dense_9/CastCastdense_9/BiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Castc
dense_9/RealRealdense_9/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Realn
dense_9/ExpExpdense_9/Real:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Expc
dense_9/ImagImagdense_9/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Imagr
dense_9/Exp_1Expdense_9/Imag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Exp_1n
dense_9/IsNanIsNandense_9/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNanq
dense_9/ones_like/ShapeShapedense_9/Exp:y:0*
T0*
_output_shapes
:2
dense_9/ones_like/Shapew
dense_9/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like/ConstЈ
dense_9/ones_likeFill dense_9/ones_like/Shape:output:0 dense_9/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_likeІ
dense_9/SelectV2SelectV2dense_9/IsNan:y:0dense_9/ones_like:output:0dense_9/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2t
dense_9/IsNan_1IsNandense_9/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNan_1w
dense_9/ones_like_1/ShapeShapedense_9/Exp_1:y:0*
T0*
_output_shapes
:2
dense_9/ones_like_1/Shape{
dense_9/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like_1/ConstА
dense_9/ones_like_1Fill"dense_9/ones_like_1/Shape:output:0"dense_9/ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_like_1А
dense_9/SelectV2_1SelectV2dense_9/IsNan_1:y:0dense_9/ones_like_1:output:0dense_9/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2_1
dense_9/mulMuldense_9/SelectV2:output:0dense_9/SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/mulr
dense_9/IsNan_2IsNandense_9/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNan_2u
dense_9/ones_like_2/ShapeShapedense_9/mul:z:0*
T0*
_output_shapes
:2
dense_9/ones_like_2/Shape{
dense_9/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like_2/ConstА
dense_9/ones_like_2Fill"dense_9/ones_like_2/Shape:output:0"dense_9/ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_like_2Ў
dense_9/SelectV2_2SelectV2dense_9/IsNan_2:y:0dense_9/ones_like_2:output:0dense_9/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2_2
dropout_3/IdentityIdentitydense_9/SelectV2_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout_3/IdentityБ
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/free
dense_10/Tensordot/ShapeShapedropout_3/Identity:output:0*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axisў
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2_1~
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/ConstЄ
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1Ќ
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axisн
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concatА
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stackР
dense_10/Tensordot/transpose	Transposedropout_3/Identity:output:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Tensordot/transposeУ
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_10/Tensordot/ReshapeТ
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_10/Tensordot/MatMul
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/Const_2
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisъ
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1Д
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/TensordotЇ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpЋ
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/BiasAdd
dense_10/CastCastdense_10/BiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Castf
dense_10/RealRealdense_10/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Realq
dense_10/ExpExpdense_10/Real:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Expf
dense_10/ImagImagdense_10/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Imagu
dense_10/Exp_1Expdense_10/Imag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Exp_1q
dense_10/IsNanIsNandense_10/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNant
dense_10/ones_like/ShapeShapedense_10/Exp:y:0*
T0*
_output_shapes
:2
dense_10/ones_like/Shapey
dense_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like/ConstЌ
dense_10/ones_likeFill!dense_10/ones_like/Shape:output:0!dense_10/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_likeЋ
dense_10/SelectV2SelectV2dense_10/IsNan:y:0dense_10/ones_like:output:0dense_10/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2w
dense_10/IsNan_1IsNandense_10/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNan_1z
dense_10/ones_like_1/ShapeShapedense_10/Exp_1:y:0*
T0*
_output_shapes
:2
dense_10/ones_like_1/Shape}
dense_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like_1/ConstД
dense_10/ones_like_1Fill#dense_10/ones_like_1/Shape:output:0#dense_10/ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_like_1Е
dense_10/SelectV2_1SelectV2dense_10/IsNan_1:y:0dense_10/ones_like_1:output:0dense_10/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2_1
dense_10/mulMuldense_10/SelectV2:output:0dense_10/SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/mulu
dense_10/IsNan_2IsNandense_10/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNan_2x
dense_10/ones_like_2/ShapeShapedense_10/mul:z:0*
T0*
_output_shapes
:2
dense_10/ones_like_2/Shape}
dense_10/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like_2/ConstД
dense_10/ones_like_2Fill#dense_10/ones_like_2/Shape:output:0#dense_10/ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_like_2Г
dense_10/SelectV2_2SelectV2dense_10/IsNan_2:y:0dense_10/ones_like_2:output:0dense_10/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2_2s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ0  2
flatten_3/Const
flatten_3/ReshapeReshapedense_10/SelectV2_2:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџА2
flatten_3/ReshapeЉ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02 
dense_11/MatMul/ReadVariableOpЂ
dense_11/MatMulMatMulflatten_3/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/MatMulЇ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_11/BiasAdd/ReadVariableOpЅ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/SoftmaxЛ
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namedense_9_input
ж
Щ
H__inference_sequential_3_layer_call_and_return_conditional_losses_162510

inputs-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*dense_10_tensordot_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identityЂdense_10/BiasAdd/ReadVariableOpЂ!dense_10/Tensordot/ReadVariableOpЂdense_11/BiasAdd/ReadVariableOpЂdense_11/MatMul/ReadVariableOpЂdense_9/BiasAdd/ReadVariableOpЂ dense_9/Tensordot/ReadVariableOpЎ
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axes
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/freeh
dense_9/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_9/Tensordot/Shape
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisљ
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axisџ
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1Ј
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisи
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatЌ
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stackЈ
dense_9/Tensordot/transpose	Transposeinputs!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Tensordot/transposeП
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_9/Tensordot/ReshapeО
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_9/Tensordot/MatMul
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/Const_2
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisх
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1А
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/TensordotЄ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpЇ
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/BiasAdd
dense_9/CastCastdense_9/BiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Castc
dense_9/RealRealdense_9/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Realn
dense_9/ExpExpdense_9/Real:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Expc
dense_9/ImagImagdense_9/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Imagr
dense_9/Exp_1Expdense_9/Imag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Exp_1n
dense_9/IsNanIsNandense_9/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNanq
dense_9/ones_like/ShapeShapedense_9/Exp:y:0*
T0*
_output_shapes
:2
dense_9/ones_like/Shapew
dense_9/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like/ConstЈ
dense_9/ones_likeFill dense_9/ones_like/Shape:output:0 dense_9/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_likeІ
dense_9/SelectV2SelectV2dense_9/IsNan:y:0dense_9/ones_like:output:0dense_9/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2t
dense_9/IsNan_1IsNandense_9/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNan_1w
dense_9/ones_like_1/ShapeShapedense_9/Exp_1:y:0*
T0*
_output_shapes
:2
dense_9/ones_like_1/Shape{
dense_9/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like_1/ConstА
dense_9/ones_like_1Fill"dense_9/ones_like_1/Shape:output:0"dense_9/ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_like_1А
dense_9/SelectV2_1SelectV2dense_9/IsNan_1:y:0dense_9/ones_like_1:output:0dense_9/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2_1
dense_9/mulMuldense_9/SelectV2:output:0dense_9/SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/mulr
dense_9/IsNan_2IsNandense_9/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNan_2u
dense_9/ones_like_2/ShapeShapedense_9/mul:z:0*
T0*
_output_shapes
:2
dense_9/ones_like_2/Shape{
dense_9/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like_2/ConstА
dense_9/ones_like_2Fill"dense_9/ones_like_2/Shape:output:0"dense_9/ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_like_2Ў
dense_9/SelectV2_2SelectV2dense_9/IsNan_2:y:0dense_9/ones_like_2:output:0dense_9/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2_2
dropout_3/IdentityIdentitydense_9/SelectV2_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout_3/IdentityБ
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/free
dense_10/Tensordot/ShapeShapedropout_3/Identity:output:0*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axisў
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2_1~
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/ConstЄ
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1Ќ
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axisн
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concatА
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stackР
dense_10/Tensordot/transpose	Transposedropout_3/Identity:output:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Tensordot/transposeУ
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_10/Tensordot/ReshapeТ
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_10/Tensordot/MatMul
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/Const_2
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisъ
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1Д
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/TensordotЇ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpЋ
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/BiasAdd
dense_10/CastCastdense_10/BiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Castf
dense_10/RealRealdense_10/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Realq
dense_10/ExpExpdense_10/Real:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Expf
dense_10/ImagImagdense_10/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Imagu
dense_10/Exp_1Expdense_10/Imag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Exp_1q
dense_10/IsNanIsNandense_10/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNant
dense_10/ones_like/ShapeShapedense_10/Exp:y:0*
T0*
_output_shapes
:2
dense_10/ones_like/Shapey
dense_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like/ConstЌ
dense_10/ones_likeFill!dense_10/ones_like/Shape:output:0!dense_10/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_likeЋ
dense_10/SelectV2SelectV2dense_10/IsNan:y:0dense_10/ones_like:output:0dense_10/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2w
dense_10/IsNan_1IsNandense_10/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNan_1z
dense_10/ones_like_1/ShapeShapedense_10/Exp_1:y:0*
T0*
_output_shapes
:2
dense_10/ones_like_1/Shape}
dense_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like_1/ConstД
dense_10/ones_like_1Fill#dense_10/ones_like_1/Shape:output:0#dense_10/ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_like_1Е
dense_10/SelectV2_1SelectV2dense_10/IsNan_1:y:0dense_10/ones_like_1:output:0dense_10/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2_1
dense_10/mulMuldense_10/SelectV2:output:0dense_10/SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/mulu
dense_10/IsNan_2IsNandense_10/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNan_2x
dense_10/ones_like_2/ShapeShapedense_10/mul:z:0*
T0*
_output_shapes
:2
dense_10/ones_like_2/Shape}
dense_10/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like_2/ConstД
dense_10/ones_like_2Fill#dense_10/ones_like_2/Shape:output:0#dense_10/ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_like_2Г
dense_10/SelectV2_2SelectV2dense_10/IsNan_2:y:0dense_10/ones_like_2:output:0dense_10/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2_2s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ0  2
flatten_3/Const
flatten_3/ReshapeReshapedense_10/SelectV2_2:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџА2
flatten_3/ReshapeЉ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02 
dense_11/MatMul/ReadVariableOpЂ
dense_11/MatMulMatMulflatten_3/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/MatMulЇ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_11/BiasAdd/ReadVariableOpЅ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/SoftmaxЛ
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
кЅ
Х
!__inference__wrapped_model_161952
dense_9_input:
6sequential_3_dense_9_tensordot_readvariableop_resource8
4sequential_3_dense_9_biasadd_readvariableop_resource;
7sequential_3_dense_10_tensordot_readvariableop_resource9
5sequential_3_dense_10_biasadd_readvariableop_resource8
4sequential_3_dense_11_matmul_readvariableop_resource9
5sequential_3_dense_11_biasadd_readvariableop_resource
identityЂ,sequential_3/dense_10/BiasAdd/ReadVariableOpЂ.sequential_3/dense_10/Tensordot/ReadVariableOpЂ,sequential_3/dense_11/BiasAdd/ReadVariableOpЂ+sequential_3/dense_11/MatMul/ReadVariableOpЂ+sequential_3/dense_9/BiasAdd/ReadVariableOpЂ-sequential_3/dense_9/Tensordot/ReadVariableOpе
-sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02/
-sequential_3/dense_9/Tensordot/ReadVariableOp
#sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_3/dense_9/Tensordot/axes
#sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_3/dense_9/Tensordot/free
$sequential_3/dense_9/Tensordot/ShapeShapedense_9_input*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/Shape
,sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/GatherV2/axisК
'sequential_3/dense_9/Tensordot/GatherV2GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/free:output:05sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/GatherV2Ђ
.sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_3/dense_9/Tensordot/GatherV2_1/axisР
)sequential_3/dense_9/Tensordot/GatherV2_1GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/axes:output:07sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_3/dense_9/Tensordot/GatherV2_1
$sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_3/dense_9/Tensordot/Constд
#sequential_3/dense_9/Tensordot/ProdProd0sequential_3/dense_9/Tensordot/GatherV2:output:0-sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_3/dense_9/Tensordot/Prod
&sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_3/dense_9/Tensordot/Const_1м
%sequential_3/dense_9/Tensordot/Prod_1Prod2sequential_3/dense_9/Tensordot/GatherV2_1:output:0/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_3/dense_9/Tensordot/Prod_1
*sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_3/dense_9/Tensordot/concat/axis
%sequential_3/dense_9/Tensordot/concatConcatV2,sequential_3/dense_9/Tensordot/free:output:0,sequential_3/dense_9/Tensordot/axes:output:03sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_9/Tensordot/concatр
$sequential_3/dense_9/Tensordot/stackPack,sequential_3/dense_9/Tensordot/Prod:output:0.sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/stackж
(sequential_3/dense_9/Tensordot/transpose	Transposedense_9_input.sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2*
(sequential_3/dense_9/Tensordot/transposeѓ
&sequential_3/dense_9/Tensordot/ReshapeReshape,sequential_3/dense_9/Tensordot/transpose:y:0-sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2(
&sequential_3/dense_9/Tensordot/Reshapeђ
%sequential_3/dense_9/Tensordot/MatMulMatMul/sequential_3/dense_9/Tensordot/Reshape:output:05sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2'
%sequential_3/dense_9/Tensordot/MatMul
&sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_3/dense_9/Tensordot/Const_2
,sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/concat_1/axisІ
'sequential_3/dense_9/Tensordot/concat_1ConcatV20sequential_3/dense_9/Tensordot/GatherV2:output:0/sequential_3/dense_9/Tensordot/Const_2:output:05sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/concat_1ф
sequential_3/dense_9/TensordotReshape/sequential_3/dense_9/Tensordot/MatMul:product:00sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2 
sequential_3/dense_9/TensordotЫ
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOpл
sequential_3/dense_9/BiasAddBiasAdd'sequential_3/dense_9/Tensordot:output:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/BiasAddЊ
sequential_3/dense_9/CastCast%sequential_3/dense_9/BiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/Cast
sequential_3/dense_9/RealRealsequential_3/dense_9/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/Real
sequential_3/dense_9/ExpExp"sequential_3/dense_9/Real:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/Exp
sequential_3/dense_9/ImagImagsequential_3/dense_9/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/Imag
sequential_3/dense_9/Exp_1Exp"sequential_3/dense_9/Imag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/Exp_1
sequential_3/dense_9/IsNanIsNansequential_3/dense_9/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/IsNan
$sequential_3/dense_9/ones_like/ShapeShapesequential_3/dense_9/Exp:y:0*
T0*
_output_shapes
:2&
$sequential_3/dense_9/ones_like/Shape
$sequential_3/dense_9/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_3/dense_9/ones_like/Constм
sequential_3/dense_9/ones_likeFill-sequential_3/dense_9/ones_like/Shape:output:0-sequential_3/dense_9/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2 
sequential_3/dense_9/ones_likeч
sequential_3/dense_9/SelectV2SelectV2sequential_3/dense_9/IsNan:y:0'sequential_3/dense_9/ones_like:output:0sequential_3/dense_9/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/SelectV2
sequential_3/dense_9/IsNan_1IsNansequential_3/dense_9/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/IsNan_1
&sequential_3/dense_9/ones_like_1/ShapeShapesequential_3/dense_9/Exp_1:y:0*
T0*
_output_shapes
:2(
&sequential_3/dense_9/ones_like_1/Shape
&sequential_3/dense_9/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_3/dense_9/ones_like_1/Constф
 sequential_3/dense_9/ones_like_1Fill/sequential_3/dense_9/ones_like_1/Shape:output:0/sequential_3/dense_9/ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2"
 sequential_3/dense_9/ones_like_1ё
sequential_3/dense_9/SelectV2_1SelectV2 sequential_3/dense_9/IsNan_1:y:0)sequential_3/dense_9/ones_like_1:output:0sequential_3/dense_9/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2!
sequential_3/dense_9/SelectV2_1У
sequential_3/dense_9/mulMul&sequential_3/dense_9/SelectV2:output:0(sequential_3/dense_9/SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/mul
sequential_3/dense_9/IsNan_2IsNansequential_3/dense_9/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_9/IsNan_2
&sequential_3/dense_9/ones_like_2/ShapeShapesequential_3/dense_9/mul:z:0*
T0*
_output_shapes
:2(
&sequential_3/dense_9/ones_like_2/Shape
&sequential_3/dense_9/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_3/dense_9/ones_like_2/Constф
 sequential_3/dense_9/ones_like_2Fill/sequential_3/dense_9/ones_like_2/Shape:output:0/sequential_3/dense_9/ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2"
 sequential_3/dense_9/ones_like_2я
sequential_3/dense_9/SelectV2_2SelectV2 sequential_3/dense_9/IsNan_2:y:0)sequential_3/dense_9/ones_like_2:output:0sequential_3/dense_9/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2!
sequential_3/dense_9/SelectV2_2Ў
sequential_3/dropout_3/IdentityIdentity(sequential_3/dense_9/SelectV2_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2!
sequential_3/dropout_3/Identityи
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOp
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axes
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/freeІ
%sequential_3/dense_10/Tensordot/ShapeShape(sequential_3/dropout_3/Identity:output:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape 
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axisП
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2Є
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axisХ
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/Constи
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/Prod
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1р
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axis
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concatф
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stackє
)sequential_3/dense_10/Tensordot/transpose	Transpose(sequential_3/dropout_3/Identity:output:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2+
)sequential_3/dense_10/Tensordot/transposeї
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2)
'sequential_3/dense_10/Tensordot/Reshapeі
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&sequential_3/dense_10/Tensordot/MatMul
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential_3/dense_10/Tensordot/Const_2 
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axisЋ
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1ш
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2!
sequential_3/dense_10/TensordotЮ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOpп
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_10/BiasAdd­
sequential_3/dense_10/CastCast&sequential_3/dense_10/BiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_10/Cast
sequential_3/dense_10/RealRealsequential_3/dense_10/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_10/Real
sequential_3/dense_10/ExpExp#sequential_3/dense_10/Real:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_10/Exp
sequential_3/dense_10/ImagImagsequential_3/dense_10/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_10/Imag
sequential_3/dense_10/Exp_1Exp#sequential_3/dense_10/Imag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_10/Exp_1
sequential_3/dense_10/IsNanIsNansequential_3/dense_10/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_10/IsNan
%sequential_3/dense_10/ones_like/ShapeShapesequential_3/dense_10/Exp:y:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/ones_like/Shape
%sequential_3/dense_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%sequential_3/dense_10/ones_like/Constр
sequential_3/dense_10/ones_likeFill.sequential_3/dense_10/ones_like/Shape:output:0.sequential_3/dense_10/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2!
sequential_3/dense_10/ones_likeь
sequential_3/dense_10/SelectV2SelectV2sequential_3/dense_10/IsNan:y:0(sequential_3/dense_10/ones_like:output:0sequential_3/dense_10/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2 
sequential_3/dense_10/SelectV2
sequential_3/dense_10/IsNan_1IsNansequential_3/dense_10/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_10/IsNan_1Ё
'sequential_3/dense_10/ones_like_1/ShapeShapesequential_3/dense_10/Exp_1:y:0*
T0*
_output_shapes
:2)
'sequential_3/dense_10/ones_like_1/Shape
'sequential_3/dense_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'sequential_3/dense_10/ones_like_1/Constш
!sequential_3/dense_10/ones_like_1Fill0sequential_3/dense_10/ones_like_1/Shape:output:00sequential_3/dense_10/ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!sequential_3/dense_10/ones_like_1і
 sequential_3/dense_10/SelectV2_1SelectV2!sequential_3/dense_10/IsNan_1:y:0*sequential_3/dense_10/ones_like_1:output:0sequential_3/dense_10/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2"
 sequential_3/dense_10/SelectV2_1Ч
sequential_3/dense_10/mulMul'sequential_3/dense_10/SelectV2:output:0)sequential_3/dense_10/SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_10/mul
sequential_3/dense_10/IsNan_2IsNansequential_3/dense_10/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
sequential_3/dense_10/IsNan_2
'sequential_3/dense_10/ones_like_2/ShapeShapesequential_3/dense_10/mul:z:0*
T0*
_output_shapes
:2)
'sequential_3/dense_10/ones_like_2/Shape
'sequential_3/dense_10/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'sequential_3/dense_10/ones_like_2/Constш
!sequential_3/dense_10/ones_like_2Fill0sequential_3/dense_10/ones_like_2/Shape:output:00sequential_3/dense_10/ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!sequential_3/dense_10/ones_like_2є
 sequential_3/dense_10/SelectV2_2SelectV2!sequential_3/dense_10/IsNan_2:y:0*sequential_3/dense_10/ones_like_2:output:0sequential_3/dense_10/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2"
 sequential_3/dense_10/SelectV2_2
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ0  2
sequential_3/flatten_3/Constа
sequential_3/flatten_3/ReshapeReshape)sequential_3/dense_10/SelectV2_2:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџА2 
sequential_3/flatten_3/Reshapeа
+sequential_3/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02-
+sequential_3/dense_11/MatMul/ReadVariableOpж
sequential_3/dense_11/MatMulMatMul'sequential_3/flatten_3/Reshape:output:03sequential_3/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
sequential_3/dense_11/MatMulЮ
,sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_3/dense_11/BiasAdd/ReadVariableOpй
sequential_3/dense_11/BiasAddBiasAdd&sequential_3/dense_11/MatMul:product:04sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
sequential_3/dense_11/BiasAddЃ
sequential_3/dense_11/SoftmaxSoftmax&sequential_3/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
sequential_3/dense_11/Softmax
IdentityIdentity'sequential_3/dense_11/Softmax:softmax:0-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp-^sequential_3/dense_11/BiasAdd/ReadVariableOp,^sequential_3/dense_11/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp.^sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2`
.sequential_3/dense_10/Tensordot/ReadVariableOp.sequential_3/dense_10/Tensordot/ReadVariableOp2\
,sequential_3/dense_11/BiasAdd/ReadVariableOp,sequential_3/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_11/MatMul/ReadVariableOp+sequential_3/dense_11/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2^
-sequential_3/dense_9/Tensordot/ReadVariableOp-sequential_3/dense_9/Tensordot/ReadVariableOp:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namedense_9_input
Є
F
*__inference_dropout_3_layer_call_fn_162888

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1620402
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_162954

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ0  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџА2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџА2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ
а
H__inference_sequential_3_layer_call_and_return_conditional_losses_162659
dense_9_input-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*dense_10_tensordot_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identityЂdense_10/BiasAdd/ReadVariableOpЂ!dense_10/Tensordot/ReadVariableOpЂdense_11/BiasAdd/ReadVariableOpЂdense_11/MatMul/ReadVariableOpЂdense_9/BiasAdd/ReadVariableOpЂ dense_9/Tensordot/ReadVariableOpЎ
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axes
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/freeo
dense_9/Tensordot/ShapeShapedense_9_input*
T0*
_output_shapes
:2
dense_9/Tensordot/Shape
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisљ
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axisџ
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1Ј
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisи
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatЌ
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stackЏ
dense_9/Tensordot/transpose	Transposedense_9_input!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Tensordot/transposeП
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_9/Tensordot/ReshapeО
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_9/Tensordot/MatMul
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/Const_2
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisх
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1А
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/TensordotЄ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpЇ
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/BiasAdd
dense_9/CastCastdense_9/BiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Castc
dense_9/RealRealdense_9/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Realn
dense_9/ExpExpdense_9/Real:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Expc
dense_9/ImagImagdense_9/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Imagr
dense_9/Exp_1Expdense_9/Imag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/Exp_1n
dense_9/IsNanIsNandense_9/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNanq
dense_9/ones_like/ShapeShapedense_9/Exp:y:0*
T0*
_output_shapes
:2
dense_9/ones_like/Shapew
dense_9/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like/ConstЈ
dense_9/ones_likeFill dense_9/ones_like/Shape:output:0 dense_9/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_likeІ
dense_9/SelectV2SelectV2dense_9/IsNan:y:0dense_9/ones_like:output:0dense_9/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2t
dense_9/IsNan_1IsNandense_9/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNan_1w
dense_9/ones_like_1/ShapeShapedense_9/Exp_1:y:0*
T0*
_output_shapes
:2
dense_9/ones_like_1/Shape{
dense_9/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like_1/ConstА
dense_9/ones_like_1Fill"dense_9/ones_like_1/Shape:output:0"dense_9/ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_like_1А
dense_9/SelectV2_1SelectV2dense_9/IsNan_1:y:0dense_9/ones_like_1:output:0dense_9/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2_1
dense_9/mulMuldense_9/SelectV2:output:0dense_9/SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/mulr
dense_9/IsNan_2IsNandense_9/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/IsNan_2u
dense_9/ones_like_2/ShapeShapedense_9/mul:z:0*
T0*
_output_shapes
:2
dense_9/ones_like_2/Shape{
dense_9/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_9/ones_like_2/ConstА
dense_9/ones_like_2Fill"dense_9/ones_like_2/Shape:output:0"dense_9/ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/ones_like_2Ў
dense_9/SelectV2_2SelectV2dense_9/IsNan_2:y:0dense_9/ones_like_2:output:0dense_9/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_9/SelectV2_2w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout_3/dropout/ConstЊ
dropout_3/dropout/MulMuldense_9/SelectV2_2:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/Mul}
dropout_3/dropout/ShapeShapedense_9/SelectV2_2:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeж
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dropout_3/dropout/GreaterEqual/yъ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2 
dropout_3/dropout/GreaterEqualЁ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/CastІ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/Mul_1Б
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/free
dense_10/Tensordot/ShapeShapedropout_3/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axisў
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2_1~
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/ConstЄ
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1Ќ
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axisн
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concatА
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stackР
dense_10/Tensordot/transpose	Transposedropout_3/dropout/Mul_1:z:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Tensordot/transposeУ
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_10/Tensordot/ReshapeТ
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_10/Tensordot/MatMul
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/Const_2
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisъ
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1Д
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/TensordotЇ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpЋ
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/BiasAdd
dense_10/CastCastdense_10/BiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Castf
dense_10/RealRealdense_10/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Realq
dense_10/ExpExpdense_10/Real:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Expf
dense_10/ImagImagdense_10/Cast:y:0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Imagu
dense_10/Exp_1Expdense_10/Imag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/Exp_1q
dense_10/IsNanIsNandense_10/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNant
dense_10/ones_like/ShapeShapedense_10/Exp:y:0*
T0*
_output_shapes
:2
dense_10/ones_like/Shapey
dense_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like/ConstЌ
dense_10/ones_likeFill!dense_10/ones_like/Shape:output:0!dense_10/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_likeЋ
dense_10/SelectV2SelectV2dense_10/IsNan:y:0dense_10/ones_like:output:0dense_10/Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2w
dense_10/IsNan_1IsNandense_10/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNan_1z
dense_10/ones_like_1/ShapeShapedense_10/Exp_1:y:0*
T0*
_output_shapes
:2
dense_10/ones_like_1/Shape}
dense_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like_1/ConstД
dense_10/ones_like_1Fill#dense_10/ones_like_1/Shape:output:0#dense_10/ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_like_1Е
dense_10/SelectV2_1SelectV2dense_10/IsNan_1:y:0dense_10/ones_like_1:output:0dense_10/Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2_1
dense_10/mulMuldense_10/SelectV2:output:0dense_10/SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/mulu
dense_10/IsNan_2IsNandense_10/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/IsNan_2x
dense_10/ones_like_2/ShapeShapedense_10/mul:z:0*
T0*
_output_shapes
:2
dense_10/ones_like_2/Shape}
dense_10/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_10/ones_like_2/ConstД
dense_10/ones_like_2Fill#dense_10/ones_like_2/Shape:output:0#dense_10/ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/ones_like_2Г
dense_10/SelectV2_2SelectV2dense_10/IsNan_2:y:0dense_10/ones_like_2:output:0dense_10/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_10/SelectV2_2s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ0  2
flatten_3/Const
flatten_3/ReshapeReshapedense_10/SelectV2_2:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџА2
flatten_3/ReshapeЉ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02 
dense_11/MatMul/ReadVariableOpЂ
dense_11/MatMulMatMulflatten_3/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/MatMulЇ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_11/BiasAdd/ReadVariableOpЅ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_11/SoftmaxЛ
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namedense_9_input
ь
я
H__inference_sequential_3_layer_call_and_return_conditional_losses_162207

inputs
dense_9_162189
dense_9_162191
dense_10_162195
dense_10_162197
dense_11_162201
dense_11_162203
identityЂ dense_10/StatefulPartitionedCallЂ dense_11/StatefulPartitionedCallЂdense_9/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCall
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_162189dense_9_162191*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_1620072!
dense_9/StatefulPartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1620352#
!dropout_3/StatefulPartitionedCallМ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_10_162195dense_10_162197*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_1621042"
 dense_10/StatefulPartitionedCallћ
flatten_3/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1621262
flatten_3/PartitionedCallА
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_11_162201dense_11_162203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_1621452"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
шr
о
"__inference__traced_restore_163174
file_prefix#
assignvariableop_dense_9_kernel#
assignvariableop_1_dense_9_bias&
"assignvariableop_2_dense_10_kernel$
 assignvariableop_3_dense_10_bias&
"assignvariableop_4_dense_11_kernel$
 assignvariableop_5_dense_11_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_1-
)assignvariableop_15_adam_dense_9_kernel_m+
'assignvariableop_16_adam_dense_9_bias_m.
*assignvariableop_17_adam_dense_10_kernel_m,
(assignvariableop_18_adam_dense_10_bias_m.
*assignvariableop_19_adam_dense_11_kernel_m,
(assignvariableop_20_adam_dense_11_bias_m-
)assignvariableop_21_adam_dense_9_kernel_v+
'assignvariableop_22_adam_dense_9_bias_v.
*assignvariableop_23_adam_dense_10_kernel_v,
(assignvariableop_24_adam_dense_10_bias_v.
*assignvariableop_25_adam_dense_11_kernel_v,
(assignvariableop_26_adam_dense_11_bias_v
identity_28ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЦ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesИ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Є
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_9_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_10_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_10_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_11_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_11_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6Ё
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѓ
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ђ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ў
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ё
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ё
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ѓ
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ѓ
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Б
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_9_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Џ
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_9_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17В
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_10_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18А
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_10_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19В
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_11_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20А
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_11_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Б
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_9_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Џ
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_9_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23В
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_10_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24А
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_10_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25В
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_11_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26А
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_11_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpА
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27Ѓ
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*
_input_shapesp
n: :::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
э
О
-__inference_sequential_3_layer_call_fn_162527

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1622072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_162873

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeИ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
dropout/GreaterEqual/yТ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
}
(__inference_dense_9_layer_call_fn_162861

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_1620072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э
О
-__inference_sequential_3_layer_call_fn_162544

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1622452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_162126

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ0  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџА2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџА2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д1
т
C__inference_dense_9_layer_call_and_return_conditional_losses_162007

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
BiasAddk
CastCastBiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
CastK
RealRealCast:y:0*+
_output_shapes
:џџџџџџџџџ2
RealV
ExpExpReal:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ExpK
ImagImagCast:y:0*+
_output_shapes
:џџџџџџџџџ2
ImagZ
Exp_1ExpImag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Exp_1V
IsNanIsNanExp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
IsNanY
ones_like/ShapeShapeExp:y:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	ones_like~
SelectV2SelectV2	IsNan:y:0ones_like:output:0Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2\
IsNan_1IsNan	Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
IsNan_1_
ones_like_1/ShapeShape	Exp_1:y:0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ones_like_1

SelectV2_1SelectV2IsNan_1:y:0ones_like_1:output:0	Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2_1o
mulMulSelectV2:output:0SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
mulZ
IsNan_2IsNanmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
IsNan_2]
ones_like_2/ShapeShapemul:z:0*
T0*
_output_shapes
:2
ones_like_2/Shapek
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_2/Const
ones_like_2Fillones_like_2/Shape:output:0ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ones_like_2

SelectV2_2SelectV2IsNan_2:y:0ones_like_2:output:0mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2_2
IdentityIdentitySelectV2_2:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_162035

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeИ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
dropout/GreaterEqual/yТ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Х
-__inference_sequential_3_layer_call_fn_162801
dense_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCalldense_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1622452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namedense_9_input
и
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_162878

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о
~
)__inference_dense_11_layer_call_fn_162979

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_1621452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџА::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџА
 
_user_specified_nameinputs
љ	
н
D__inference_dense_11_layer_call_and_return_conditional_losses_162970

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџА::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџА
 
_user_specified_nameinputs
и
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_162040

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е1
у
D__inference_dense_10_layer_call_and_return_conditional_losses_162104

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
BiasAddk
CastCastBiasAdd:output:0*

DstT0*

SrcT0*+
_output_shapes
:џџџџџџџџџ2
CastK
RealRealCast:y:0*+
_output_shapes
:џџџџџџџџџ2
RealV
ExpExpReal:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ExpK
ImagImagCast:y:0*+
_output_shapes
:џџџџџџџџџ2
ImagZ
Exp_1ExpImag:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Exp_1V
IsNanIsNanExp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2
IsNanY
ones_like/ShapeShapeExp:y:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	ones_like~
SelectV2SelectV2	IsNan:y:0ones_like:output:0Exp:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2\
IsNan_1IsNan	Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
IsNan_1_
ones_like_1/ShapeShape	Exp_1:y:0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ones_like_1

SelectV2_1SelectV2IsNan_1:y:0ones_like_1:output:0	Exp_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2_1o
mulMulSelectV2:output:0SelectV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
mulZ
IsNan_2IsNanmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
IsNan_2]
ones_like_2/ShapeShapemul:z:0*
T0*
_output_shapes
:2
ones_like_2/Shapek
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_2/Const
ones_like_2Fillones_like_2/Shape:output:0ones_like_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ones_like_2

SelectV2_2SelectV2IsNan_2:y:0ones_like_2:output:0mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ2

SelectV2_2
IdentityIdentitySelectV2_2:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Х
-__inference_sequential_3_layer_call_fn_162784
dense_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCalldense_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1622072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namedense_9_input
ь
~
)__inference_dense_10_layer_call_fn_162948

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_1621042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Л
serving_defaultЇ
K
dense_9_input:
serving_default_dense_9_input:0џџџџџџџџџ<
dense_110
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:џЋ
ђ'
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
`_default_save_signature
*a&call_and_return_all_conditional_losses
b__call__"%
_tf_keras_sequentialљ${"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_9_input"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 20, "activation": "mul_c2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 20, "activation": "mul_c2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 28}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_9_input"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 20, "activation": "mul_c2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 20, "activation": "mul_c2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
є

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 20, "activation": "mul_c2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 28}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 28, 28]}}
ц
regularization_losses
	variables
trainable_variables
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"з
_tf_keras_layerН{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}
і

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"б
_tf_keras_layerЗ{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 20, "activation": "mul_c2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 28, 20]}}
ц
regularization_losses
	variables
trainable_variables
	keras_api
*i&call_and_return_all_conditional_losses
j__call__"з
_tf_keras_layerН{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ѕ

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
*k&call_and_return_all_conditional_losses
l__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 560}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 560]}}
П
&iter

'beta_1

(beta_2
	)decay
*learning_ratemTmUmVmW mX!mYvZv[v\v] v^!v_"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
Ъ
regularization_losses
+layer_regularization_losses
,non_trainable_variables
	variables

-layers
	trainable_variables
.layer_metrics
/metrics
b__call__
`_default_save_signature
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
,
mserving_default"
signature_map
 :2dense_9/kernel
:2dense_9/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
0layer_regularization_losses
1non_trainable_variables
	variables

2layers
trainable_variables
3layer_metrics
4metrics
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses
5layer_regularization_losses
6non_trainable_variables
	variables

7layers
trainable_variables
8layer_metrics
9metrics
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
!:2dense_10/kernel
:2dense_10/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
:layer_regularization_losses
;non_trainable_variables
	variables

<layers
trainable_variables
=layer_metrics
>metrics
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses
?layer_regularization_losses
@non_trainable_variables
	variables

Alayers
trainable_variables
Blayer_metrics
Cmetrics
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
": 	А
2dense_11/kernel
:
2dense_11/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
­
"regularization_losses
Dlayer_regularization_losses
Enon_trainable_variables
#	variables

Flayers
$trainable_variables
Glayer_metrics
Hmetrics
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Л
	Ktotal
	Lcount
M	variables
N	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api"П
_tf_keras_metricЄ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
K0
L1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
%:#2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
&:$2Adam/dense_10/kernel/m
 :2Adam/dense_10/bias/m
':%	А
2Adam/dense_11/kernel/m
 :
2Adam/dense_11/bias/m
%:#2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
&:$2Adam/dense_10/kernel/v
 :2Adam/dense_10/bias/v
':%	А
2Adam/dense_11/kernel/v
 :
2Adam/dense_11/bias/v
щ2ц
!__inference__wrapped_model_161952Р
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *0Ђ-
+(
dense_9_inputџџџџџџџџџ
ю2ы
H__inference_sequential_3_layer_call_and_return_conditional_losses_162659
H__inference_sequential_3_layer_call_and_return_conditional_losses_162767
H__inference_sequential_3_layer_call_and_return_conditional_losses_162510
H__inference_sequential_3_layer_call_and_return_conditional_losses_162402Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2џ
-__inference_sequential_3_layer_call_fn_162544
-__inference_sequential_3_layer_call_fn_162784
-__inference_sequential_3_layer_call_fn_162801
-__inference_sequential_3_layer_call_fn_162527Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
э2ъ
C__inference_dense_9_layer_call_and_return_conditional_losses_162852Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_9_layer_call_fn_162861Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
E__inference_dropout_3_layer_call_and_return_conditional_losses_162878
E__inference_dropout_3_layer_call_and_return_conditional_losses_162873Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_dropout_3_layer_call_fn_162883
*__inference_dropout_3_layer_call_fn_162888Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
D__inference_dense_10_layer_call_and_return_conditional_losses_162939Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_10_layer_call_fn_162948Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_flatten_3_layer_call_and_return_conditional_losses_162954Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_flatten_3_layer_call_fn_162959Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_11_layer_call_and_return_conditional_losses_162970Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_11_layer_call_fn_162979Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
бBЮ
$__inference_signature_wrapper_162287dense_9_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
!__inference__wrapped_model_161952y !:Ђ7
0Ђ-
+(
dense_9_inputџџџџџџџџџ
Њ "3Њ0
.
dense_11"
dense_11џџџџџџџџџ
Ќ
D__inference_dense_10_layer_call_and_return_conditional_losses_162939d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
)__inference_dense_10_layer_call_fn_162948W3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЅ
D__inference_dense_11_layer_call_and_return_conditional_losses_162970] !0Ђ-
&Ђ#
!
inputsџџџџџџџџџА
Њ "%Ђ"

0џџџџџџџџџ

 }
)__inference_dense_11_layer_call_fn_162979P !0Ђ-
&Ђ#
!
inputsџџџџџџџџџА
Њ "џџџџџџџџџ
Ћ
C__inference_dense_9_layer_call_and_return_conditional_losses_162852d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
(__inference_dense_9_layer_call_fn_162861W3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ­
E__inference_dropout_3_layer_call_and_return_conditional_losses_162873d7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ
p
Њ ")Ђ&

0џџџџџџџџџ
 ­
E__inference_dropout_3_layer_call_and_return_conditional_losses_162878d7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ
p 
Њ ")Ђ&

0џџџџџџџџџ
 
*__inference_dropout_3_layer_call_fn_162883W7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ
*__inference_dropout_3_layer_call_fn_162888W7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџІ
E__inference_flatten_3_layer_call_and_return_conditional_losses_162954]3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџА
 ~
*__inference_flatten_3_layer_call_fn_162959P3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџАИ
H__inference_sequential_3_layer_call_and_return_conditional_losses_162402l !;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 И
H__inference_sequential_3_layer_call_and_return_conditional_losses_162510l !;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 П
H__inference_sequential_3_layer_call_and_return_conditional_losses_162659s !BЂ?
8Ђ5
+(
dense_9_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 П
H__inference_sequential_3_layer_call_and_return_conditional_losses_162767s !BЂ?
8Ђ5
+(
dense_9_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 
-__inference_sequential_3_layer_call_fn_162527_ !;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

-__inference_sequential_3_layer_call_fn_162544_ !;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

-__inference_sequential_3_layer_call_fn_162784f !BЂ?
8Ђ5
+(
dense_9_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

-__inference_sequential_3_layer_call_fn_162801f !BЂ?
8Ђ5
+(
dense_9_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
Г
$__inference_signature_wrapper_162287 !KЂH
Ђ 
AЊ>
<
dense_9_input+(
dense_9_inputџџџџџџџџџ"3Њ0
.
dense_11"
dense_11џџџџџџџџџ
