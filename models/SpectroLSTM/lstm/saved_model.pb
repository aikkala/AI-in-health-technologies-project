??*
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
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
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
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
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.6.02unknown8ů)
?
lstm/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*"
shared_namelstm/dense/kernel
y
%lstm/dense/kernel/Read/ReadVariableOpReadVariableOplstm/dense/kernel* 
_output_shapes
:
??*
dtype0
w
lstm/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namelstm/dense/bias
p
#lstm/dense/bias/Read/ReadVariableOpReadVariableOplstm/dense/bias*
_output_shapes	
:?*
dtype0
?
lstm/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_namelstm/dense_1/kernel
|
'lstm/dense_1/kernel/Read/ReadVariableOpReadVariableOplstm/dense_1/kernel*
_output_shapes
:	?*
dtype0
z
lstm/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namelstm/dense_1/bias
s
%lstm/dense_1/bias/Read/ReadVariableOpReadVariableOplstm/dense_1/bias*
_output_shapes
:*
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
?
lstm/lstm_1/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	E?*-
shared_namelstm/lstm_1/lstm_cell/kernel
?
0lstm/lstm_1/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_1/lstm_cell/kernel*
_output_shapes
:	E?*
dtype0
?
&lstm/lstm_1/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*7
shared_name(&lstm/lstm_1/lstm_cell/recurrent_kernel
?
:lstm/lstm_1/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp&lstm/lstm_1/lstm_cell/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm/lstm_1/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelstm/lstm_1/lstm_cell/bias
?
.lstm/lstm_1/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_1/lstm_cell/bias*
_output_shapes	
:?*
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
?
Adam/lstm/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameAdam/lstm/dense/kernel/m
?
,Adam/lstm/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/dense/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/lstm/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/lstm/dense/bias/m
~
*Adam/lstm/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/dense/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/lstm/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_nameAdam/lstm/dense_1/kernel/m
?
.Adam/lstm/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/dense_1/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/lstm/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/lstm/dense_1/bias/m
?
,Adam/lstm/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/dense_1/bias/m*
_output_shapes
:*
dtype0
?
#Adam/lstm/lstm_1/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	E?*4
shared_name%#Adam/lstm/lstm_1/lstm_cell/kernel/m
?
7Adam/lstm/lstm_1/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/lstm/lstm_1/lstm_cell/kernel/m*
_output_shapes
:	E?*
dtype0
?
-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*>
shared_name/-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/m
?
AAdam/lstm/lstm_1/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
!Adam/lstm/lstm_1/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/lstm/lstm_1/lstm_cell/bias/m
?
5Adam/lstm/lstm_1/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOp!Adam/lstm/lstm_1/lstm_cell/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/lstm/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameAdam/lstm/dense/kernel/v
?
,Adam/lstm/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/dense/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/lstm/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/lstm/dense/bias/v
~
*Adam/lstm/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/dense/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/lstm/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_nameAdam/lstm/dense_1/kernel/v
?
.Adam/lstm/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/dense_1/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/lstm/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/lstm/dense_1/bias/v
?
,Adam/lstm/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/dense_1/bias/v*
_output_shapes
:*
dtype0
?
#Adam/lstm/lstm_1/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	E?*4
shared_name%#Adam/lstm/lstm_1/lstm_cell/kernel/v
?
7Adam/lstm/lstm_1/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/lstm/lstm_1/lstm_cell/kernel/v*
_output_shapes
:	E?*
dtype0
?
-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*>
shared_name/-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/v
?
AAdam/lstm/lstm_1/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
!Adam/lstm/lstm_1/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/lstm/lstm_1/lstm_cell/bias/v
?
5Adam/lstm/lstm_1/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOp!Adam/lstm/lstm_1/lstm_cell/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?,
value?,B?, B?,
?
lstm

dense1

dense2
dropout
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
 	keras_api
?
!iter

"beta_1

#beta_2
	$decay
%learning_ratemXmYmZm[&m\'m](m^v_v`vavb&vc'vd(ve
1
&0
'1
(2
3
4
5
6
 
1
&0
'1
(2
3
4
5
6
?
)layer_regularization_losses
*metrics
trainable_variables
+non_trainable_variables
,layer_metrics
regularization_losses
	variables

-layers
 
?
.
state_size

&kernel
'recurrent_kernel
(bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
 

&0
'1
(2
 

&0
'1
(2
?
3layer_regularization_losses
4metrics
trainable_variables

5states
6non_trainable_variables
7layer_metrics
regularization_losses
	variables

8layers
OM
VARIABLE_VALUElstm/dense/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUElstm/dense/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
9layer_regularization_losses

:layers
;metrics
trainable_variables
<non_trainable_variables
regularization_losses
	variables
=layer_metrics
QO
VARIABLE_VALUElstm/dense_1/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUElstm/dense_1/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
>layer_regularization_losses

?layers
@metrics
trainable_variables
Anon_trainable_variables
regularization_losses
	variables
Blayer_metrics
 
 
 
?
Clayer_regularization_losses

Dlayers
Emetrics
trainable_variables
Fnon_trainable_variables
regularization_losses
	variables
Glayer_metrics
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
b`
VARIABLE_VALUElstm/lstm_1/lstm_cell/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE&lstm/lstm_1/lstm_cell/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUElstm/lstm_1/lstm_cell/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1
 
 

0
1
2
3
 

&0
'1
(2
 

&0
'1
(2
?
Jlayer_regularization_losses

Klayers
Lmetrics
/trainable_variables
Mnon_trainable_variables
0regularization_losses
1	variables
Nlayer_metrics
 
 
 
 
 

0
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
	Ototal
	Pcount
Q	variables
R	keras_api
D
	Stotal
	Tcount
U
_fn_kwargs
V	variables
W	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

Q	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

V	variables
rp
VARIABLE_VALUEAdam/lstm/dense/kernel/mDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/lstm/dense/bias/mBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/dense_1/kernel/mDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/lstm/dense_1/bias/mBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/lstm/lstm_1/lstm_cell/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/lstm/lstm_1/lstm_cell/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/dense/kernel/vDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/lstm/dense/bias/vBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/dense_1/kernel/vDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/lstm/dense_1/bias/vBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/lstm/lstm_1/lstm_cell/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/lstm/lstm_1/lstm_cell/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????E*
dtype0* 
shape:?????????E
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lstm/lstm_1/lstm_cell/kernellstm/lstm_1/lstm_cell/bias&lstm/lstm_1/lstm_cell/recurrent_kernellstm/dense/kernellstm/dense/biaslstm/dense_1/kernellstm/dense_1/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_19310
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%lstm/dense/kernel/Read/ReadVariableOp#lstm/dense/bias/Read/ReadVariableOp'lstm/dense_1/kernel/Read/ReadVariableOp%lstm/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp0lstm/lstm_1/lstm_cell/kernel/Read/ReadVariableOp:lstm/lstm_1/lstm_cell/recurrent_kernel/Read/ReadVariableOp.lstm/lstm_1/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/lstm/dense/kernel/m/Read/ReadVariableOp*Adam/lstm/dense/bias/m/Read/ReadVariableOp.Adam/lstm/dense_1/kernel/m/Read/ReadVariableOp,Adam/lstm/dense_1/bias/m/Read/ReadVariableOp7Adam/lstm/lstm_1/lstm_cell/kernel/m/Read/ReadVariableOpAAdam/lstm/lstm_1/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp5Adam/lstm/lstm_1/lstm_cell/bias/m/Read/ReadVariableOp,Adam/lstm/dense/kernel/v/Read/ReadVariableOp*Adam/lstm/dense/bias/v/Read/ReadVariableOp.Adam/lstm/dense_1/kernel/v/Read/ReadVariableOp,Adam/lstm/dense_1/bias/v/Read/ReadVariableOp7Adam/lstm/lstm_1/lstm_cell/kernel/v/Read/ReadVariableOpAAdam/lstm/lstm_1/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp5Adam/lstm/lstm_1/lstm_cell/bias/v/Read/ReadVariableOpConst*+
Tin$
"2 	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_21906
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelstm/dense/kernellstm/dense/biaslstm/dense_1/kernellstm/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/lstm_1/lstm_cell/kernel&lstm/lstm_1/lstm_cell/recurrent_kernellstm/lstm_1/lstm_cell/biastotalcounttotal_1count_1Adam/lstm/dense/kernel/mAdam/lstm/dense/bias/mAdam/lstm/dense_1/kernel/mAdam/lstm/dense_1/bias/m#Adam/lstm/lstm_1/lstm_cell/kernel/m-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/m!Adam/lstm/lstm_1/lstm_cell/bias/mAdam/lstm/dense/kernel/vAdam/lstm/dense/bias/vAdam/lstm/dense_1/kernel/vAdam/lstm/dense_1/bias/v#Adam/lstm/lstm_1/lstm_cell/kernel/v-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/v!Adam/lstm/lstm_1/lstm_cell/bias/v**
Tin#
!2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_22006??(
?D
?
__inference__traced_save_21906
file_prefix0
,savev2_lstm_dense_kernel_read_readvariableop.
*savev2_lstm_dense_bias_read_readvariableop2
.savev2_lstm_dense_1_kernel_read_readvariableop0
,savev2_lstm_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop;
7savev2_lstm_lstm_1_lstm_cell_kernel_read_readvariableopE
Asavev2_lstm_lstm_1_lstm_cell_recurrent_kernel_read_readvariableop9
5savev2_lstm_lstm_1_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_lstm_dense_kernel_m_read_readvariableop5
1savev2_adam_lstm_dense_bias_m_read_readvariableop9
5savev2_adam_lstm_dense_1_kernel_m_read_readvariableop7
3savev2_adam_lstm_dense_1_bias_m_read_readvariableopB
>savev2_adam_lstm_lstm_1_lstm_cell_kernel_m_read_readvariableopL
Hsavev2_adam_lstm_lstm_1_lstm_cell_recurrent_kernel_m_read_readvariableop@
<savev2_adam_lstm_lstm_1_lstm_cell_bias_m_read_readvariableop7
3savev2_adam_lstm_dense_kernel_v_read_readvariableop5
1savev2_adam_lstm_dense_bias_v_read_readvariableop9
5savev2_adam_lstm_dense_1_kernel_v_read_readvariableop7
3savev2_adam_lstm_dense_1_bias_v_read_readvariableopB
>savev2_adam_lstm_lstm_1_lstm_cell_kernel_v_read_readvariableopL
Hsavev2_adam_lstm_lstm_1_lstm_cell_recurrent_kernel_v_read_readvariableop@
<savev2_adam_lstm_lstm_1_lstm_cell_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_lstm_dense_kernel_read_readvariableop*savev2_lstm_dense_bias_read_readvariableop.savev2_lstm_dense_1_kernel_read_readvariableop,savev2_lstm_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop7savev2_lstm_lstm_1_lstm_cell_kernel_read_readvariableopAsavev2_lstm_lstm_1_lstm_cell_recurrent_kernel_read_readvariableop5savev2_lstm_lstm_1_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_lstm_dense_kernel_m_read_readvariableop1savev2_adam_lstm_dense_bias_m_read_readvariableop5savev2_adam_lstm_dense_1_kernel_m_read_readvariableop3savev2_adam_lstm_dense_1_bias_m_read_readvariableop>savev2_adam_lstm_lstm_1_lstm_cell_kernel_m_read_readvariableopHsavev2_adam_lstm_lstm_1_lstm_cell_recurrent_kernel_m_read_readvariableop<savev2_adam_lstm_lstm_1_lstm_cell_bias_m_read_readvariableop3savev2_adam_lstm_dense_kernel_v_read_readvariableop1savev2_adam_lstm_dense_bias_v_read_readvariableop5savev2_adam_lstm_dense_1_kernel_v_read_readvariableop3savev2_adam_lstm_dense_1_bias_v_read_readvariableop>savev2_adam_lstm_lstm_1_lstm_cell_kernel_v_read_readvariableopHsavev2_adam_lstm_lstm_1_lstm_cell_recurrent_kernel_v_read_readvariableop<savev2_adam_lstm_lstm_1_lstm_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:	?:: : : : : :	E?:
??:?: : : : :
??:?:	?::	E?:
??:?:
??:?:	?::	E?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	E?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	E?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	E?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_18628

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
lstm_lstm_1_while_body_173054
0lstm_lstm_1_while_lstm_lstm_1_while_loop_counter:
6lstm_lstm_1_while_lstm_lstm_1_while_maximum_iterations!
lstm_lstm_1_while_placeholder#
lstm_lstm_1_while_placeholder_1#
lstm_lstm_1_while_placeholder_2#
lstm_lstm_1_while_placeholder_3#
lstm_lstm_1_while_placeholder_43
/lstm_lstm_1_while_lstm_lstm_1_strided_slice_1_0o
klstm_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_1_tensorarrayunstack_tensorlistfromtensor_0s
olstm_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0N
;lstm_lstm_1_while_lstm_cell_split_readvariableop_resource_0:	E?L
=lstm_lstm_1_while_lstm_cell_split_1_readvariableop_resource_0:	?I
5lstm_lstm_1_while_lstm_cell_readvariableop_resource_0:
??
lstm_lstm_1_while_identity 
lstm_lstm_1_while_identity_1 
lstm_lstm_1_while_identity_2 
lstm_lstm_1_while_identity_3 
lstm_lstm_1_while_identity_4 
lstm_lstm_1_while_identity_5 
lstm_lstm_1_while_identity_61
-lstm_lstm_1_while_lstm_lstm_1_strided_slice_1m
ilstm_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_1_tensorarrayunstack_tensorlistfromtensorq
mlstm_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_lstm_1_tensorarrayunstack_1_tensorlistfromtensorL
9lstm_lstm_1_while_lstm_cell_split_readvariableop_resource:	E?J
;lstm_lstm_1_while_lstm_cell_split_1_readvariableop_resource:	?G
3lstm_lstm_1_while_lstm_cell_readvariableop_resource:
????*lstm/lstm_1/while/lstm_cell/ReadVariableOp?,lstm/lstm_1/while/lstm_cell/ReadVariableOp_1?,lstm/lstm_1/while/lstm_cell/ReadVariableOp_2?,lstm/lstm_1/while/lstm_cell/ReadVariableOp_3?0lstm/lstm_1/while/lstm_cell/split/ReadVariableOp?2lstm/lstm_1/while/lstm_cell/split_1/ReadVariableOp?
Clstm/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   2E
Clstm/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
5lstm/lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemklstm_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_lstm_1_while_placeholderLlstm/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype027
5lstm/lstm_1/while/TensorArrayV2Read/TensorListGetItem?
Elstm/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
Elstm/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
7lstm/lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemolstm_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0lstm_lstm_1_while_placeholderNlstm/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
29
7lstm/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem?
+lstm/lstm_1/while/lstm_cell/ones_like/ShapeShape<lstm/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2-
+lstm/lstm_1/while/lstm_cell/ones_like/Shape?
+lstm/lstm_1/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+lstm/lstm_1/while/lstm_cell/ones_like/Const?
%lstm/lstm_1/while/lstm_cell/ones_likeFill4lstm/lstm_1/while/lstm_cell/ones_like/Shape:output:04lstm/lstm_1/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2'
%lstm/lstm_1/while/lstm_cell/ones_like?
-lstm/lstm_1/while/lstm_cell/ones_like_1/ShapeShapelstm_lstm_1_while_placeholder_3*
T0*
_output_shapes
:2/
-lstm/lstm_1/while/lstm_cell/ones_like_1/Shape?
-lstm/lstm_1/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2/
-lstm/lstm_1/while/lstm_cell/ones_like_1/Const?
'lstm/lstm_1/while/lstm_cell/ones_like_1Fill6lstm/lstm_1/while/lstm_cell/ones_like_1/Shape:output:06lstm/lstm_1/while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2)
'lstm/lstm_1/while/lstm_cell/ones_like_1?
lstm/lstm_1/while/lstm_cell/mulMul<lstm/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm/lstm_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2!
lstm/lstm_1/while/lstm_cell/mul?
!lstm/lstm_1/while/lstm_cell/mul_1Mul<lstm/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm/lstm_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2#
!lstm/lstm_1/while/lstm_cell/mul_1?
!lstm/lstm_1/while/lstm_cell/mul_2Mul<lstm/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm/lstm_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2#
!lstm/lstm_1/while/lstm_cell/mul_2?
!lstm/lstm_1/while/lstm_cell/mul_3Mul<lstm/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm/lstm_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2#
!lstm/lstm_1/while/lstm_cell/mul_3?
+lstm/lstm_1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+lstm/lstm_1/while/lstm_cell/split/split_dim?
0lstm/lstm_1/while/lstm_cell/split/ReadVariableOpReadVariableOp;lstm_lstm_1_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	E?*
dtype022
0lstm/lstm_1/while/lstm_cell/split/ReadVariableOp?
!lstm/lstm_1/while/lstm_cell/splitSplit4lstm/lstm_1/while/lstm_cell/split/split_dim:output:08lstm/lstm_1/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2#
!lstm/lstm_1/while/lstm_cell/split?
"lstm/lstm_1/while/lstm_cell/MatMulMatMul#lstm/lstm_1/while/lstm_cell/mul:z:0*lstm/lstm_1/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm/lstm_1/while/lstm_cell/MatMul?
$lstm/lstm_1/while/lstm_cell/MatMul_1MatMul%lstm/lstm_1/while/lstm_cell/mul_1:z:0*lstm/lstm_1/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm/lstm_1/while/lstm_cell/MatMul_1?
$lstm/lstm_1/while/lstm_cell/MatMul_2MatMul%lstm/lstm_1/while/lstm_cell/mul_2:z:0*lstm/lstm_1/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2&
$lstm/lstm_1/while/lstm_cell/MatMul_2?
$lstm/lstm_1/while/lstm_cell/MatMul_3MatMul%lstm/lstm_1/while/lstm_cell/mul_3:z:0*lstm/lstm_1/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm/lstm_1/while/lstm_cell/MatMul_3?
-lstm/lstm_1/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-lstm/lstm_1/while/lstm_cell/split_1/split_dim?
2lstm/lstm_1/while/lstm_cell/split_1/ReadVariableOpReadVariableOp=lstm_lstm_1_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype024
2lstm/lstm_1/while/lstm_cell/split_1/ReadVariableOp?
#lstm/lstm_1/while/lstm_cell/split_1Split6lstm/lstm_1/while/lstm_cell/split_1/split_dim:output:0:lstm/lstm_1/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2%
#lstm/lstm_1/while/lstm_cell/split_1?
#lstm/lstm_1/while/lstm_cell/BiasAddBiasAdd,lstm/lstm_1/while/lstm_cell/MatMul:product:0,lstm/lstm_1/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2%
#lstm/lstm_1/while/lstm_cell/BiasAdd?
%lstm/lstm_1/while/lstm_cell/BiasAdd_1BiasAdd.lstm/lstm_1/while/lstm_cell/MatMul_1:product:0,lstm/lstm_1/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2'
%lstm/lstm_1/while/lstm_cell/BiasAdd_1?
%lstm/lstm_1/while/lstm_cell/BiasAdd_2BiasAdd.lstm/lstm_1/while/lstm_cell/MatMul_2:product:0,lstm/lstm_1/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2'
%lstm/lstm_1/while/lstm_cell/BiasAdd_2?
%lstm/lstm_1/while/lstm_cell/BiasAdd_3BiasAdd.lstm/lstm_1/while/lstm_cell/MatMul_3:product:0,lstm/lstm_1/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2'
%lstm/lstm_1/while/lstm_cell/BiasAdd_3?
!lstm/lstm_1/while/lstm_cell/mul_4Mullstm_lstm_1_while_placeholder_30lstm/lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/while/lstm_cell/mul_4?
!lstm/lstm_1/while/lstm_cell/mul_5Mullstm_lstm_1_while_placeholder_30lstm/lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/while/lstm_cell/mul_5?
!lstm/lstm_1/while/lstm_cell/mul_6Mullstm_lstm_1_while_placeholder_30lstm/lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/while/lstm_cell/mul_6?
!lstm/lstm_1/while/lstm_cell/mul_7Mullstm_lstm_1_while_placeholder_30lstm/lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/while/lstm_cell/mul_7?
*lstm/lstm_1/while/lstm_cell/ReadVariableOpReadVariableOp5lstm_lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*lstm/lstm_1/while/lstm_cell/ReadVariableOp?
/lstm/lstm_1/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/lstm/lstm_1/while/lstm_cell/strided_slice/stack?
1lstm/lstm_1/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1lstm/lstm_1/while/lstm_cell/strided_slice/stack_1?
1lstm/lstm_1/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm/lstm_1/while/lstm_cell/strided_slice/stack_2?
)lstm/lstm_1/while/lstm_cell/strided_sliceStridedSlice2lstm/lstm_1/while/lstm_cell/ReadVariableOp:value:08lstm/lstm_1/while/lstm_cell/strided_slice/stack:output:0:lstm/lstm_1/while/lstm_cell/strided_slice/stack_1:output:0:lstm/lstm_1/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2+
)lstm/lstm_1/while/lstm_cell/strided_slice?
$lstm/lstm_1/while/lstm_cell/MatMul_4MatMul%lstm/lstm_1/while/lstm_cell/mul_4:z:02lstm/lstm_1/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2&
$lstm/lstm_1/while/lstm_cell/MatMul_4?
lstm/lstm_1/while/lstm_cell/addAddV2,lstm/lstm_1/while/lstm_cell/BiasAdd:output:0.lstm/lstm_1/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2!
lstm/lstm_1/while/lstm_cell/add?
#lstm/lstm_1/while/lstm_cell/SigmoidSigmoid#lstm/lstm_1/while/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2%
#lstm/lstm_1/while/lstm_cell/Sigmoid?
,lstm/lstm_1/while/lstm_cell/ReadVariableOp_1ReadVariableOp5lstm_lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02.
,lstm/lstm_1/while/lstm_cell/ReadVariableOp_1?
1lstm/lstm_1/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       23
1lstm/lstm_1/while/lstm_cell/strided_slice_1/stack?
3lstm/lstm_1/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       25
3lstm/lstm_1/while/lstm_cell/strided_slice_1/stack_1?
3lstm/lstm_1/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3lstm/lstm_1/while/lstm_cell/strided_slice_1/stack_2?
+lstm/lstm_1/while/lstm_cell/strided_slice_1StridedSlice4lstm/lstm_1/while/lstm_cell/ReadVariableOp_1:value:0:lstm/lstm_1/while/lstm_cell/strided_slice_1/stack:output:0<lstm/lstm_1/while/lstm_cell/strided_slice_1/stack_1:output:0<lstm/lstm_1/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2-
+lstm/lstm_1/while/lstm_cell/strided_slice_1?
$lstm/lstm_1/while/lstm_cell/MatMul_5MatMul%lstm/lstm_1/while/lstm_cell/mul_5:z:04lstm/lstm_1/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2&
$lstm/lstm_1/while/lstm_cell/MatMul_5?
!lstm/lstm_1/while/lstm_cell/add_1AddV2.lstm/lstm_1/while/lstm_cell/BiasAdd_1:output:0.lstm/lstm_1/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/while/lstm_cell/add_1?
%lstm/lstm_1/while/lstm_cell/Sigmoid_1Sigmoid%lstm/lstm_1/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2'
%lstm/lstm_1/while/lstm_cell/Sigmoid_1?
!lstm/lstm_1/while/lstm_cell/mul_8Mul)lstm/lstm_1/while/lstm_cell/Sigmoid_1:y:0lstm_lstm_1_while_placeholder_4*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/while/lstm_cell/mul_8?
,lstm/lstm_1/while/lstm_cell/ReadVariableOp_2ReadVariableOp5lstm_lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02.
,lstm/lstm_1/while/lstm_cell/ReadVariableOp_2?
1lstm/lstm_1/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       23
1lstm/lstm_1/while/lstm_cell/strided_slice_2/stack?
3lstm/lstm_1/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       25
3lstm/lstm_1/while/lstm_cell/strided_slice_2/stack_1?
3lstm/lstm_1/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3lstm/lstm_1/while/lstm_cell/strided_slice_2/stack_2?
+lstm/lstm_1/while/lstm_cell/strided_slice_2StridedSlice4lstm/lstm_1/while/lstm_cell/ReadVariableOp_2:value:0:lstm/lstm_1/while/lstm_cell/strided_slice_2/stack:output:0<lstm/lstm_1/while/lstm_cell/strided_slice_2/stack_1:output:0<lstm/lstm_1/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2-
+lstm/lstm_1/while/lstm_cell/strided_slice_2?
$lstm/lstm_1/while/lstm_cell/MatMul_6MatMul%lstm/lstm_1/while/lstm_cell/mul_6:z:04lstm/lstm_1/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2&
$lstm/lstm_1/while/lstm_cell/MatMul_6?
!lstm/lstm_1/while/lstm_cell/add_2AddV2.lstm/lstm_1/while/lstm_cell/BiasAdd_2:output:0.lstm/lstm_1/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/while/lstm_cell/add_2?
 lstm/lstm_1/while/lstm_cell/TanhTanh%lstm/lstm_1/while/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2"
 lstm/lstm_1/while/lstm_cell/Tanh?
!lstm/lstm_1/while/lstm_cell/mul_9Mul'lstm/lstm_1/while/lstm_cell/Sigmoid:y:0$lstm/lstm_1/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/while/lstm_cell/mul_9?
!lstm/lstm_1/while/lstm_cell/add_3AddV2%lstm/lstm_1/while/lstm_cell/mul_8:z:0%lstm/lstm_1/while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/while/lstm_cell/add_3?
,lstm/lstm_1/while/lstm_cell/ReadVariableOp_3ReadVariableOp5lstm_lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02.
,lstm/lstm_1/while/lstm_cell/ReadVariableOp_3?
1lstm/lstm_1/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       23
1lstm/lstm_1/while/lstm_cell/strided_slice_3/stack?
3lstm/lstm_1/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3lstm/lstm_1/while/lstm_cell/strided_slice_3/stack_1?
3lstm/lstm_1/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3lstm/lstm_1/while/lstm_cell/strided_slice_3/stack_2?
+lstm/lstm_1/while/lstm_cell/strided_slice_3StridedSlice4lstm/lstm_1/while/lstm_cell/ReadVariableOp_3:value:0:lstm/lstm_1/while/lstm_cell/strided_slice_3/stack:output:0<lstm/lstm_1/while/lstm_cell/strided_slice_3/stack_1:output:0<lstm/lstm_1/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2-
+lstm/lstm_1/while/lstm_cell/strided_slice_3?
$lstm/lstm_1/while/lstm_cell/MatMul_7MatMul%lstm/lstm_1/while/lstm_cell/mul_7:z:04lstm/lstm_1/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2&
$lstm/lstm_1/while/lstm_cell/MatMul_7?
!lstm/lstm_1/while/lstm_cell/add_4AddV2.lstm/lstm_1/while/lstm_cell/BiasAdd_3:output:0.lstm/lstm_1/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/while/lstm_cell/add_4?
%lstm/lstm_1/while/lstm_cell/Sigmoid_2Sigmoid%lstm/lstm_1/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2'
%lstm/lstm_1/while/lstm_cell/Sigmoid_2?
"lstm/lstm_1/while/lstm_cell/Tanh_1Tanh%lstm/lstm_1/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2$
"lstm/lstm_1/while/lstm_cell/Tanh_1?
"lstm/lstm_1/while/lstm_cell/mul_10Mul)lstm/lstm_1/while/lstm_cell/Sigmoid_2:y:0&lstm/lstm_1/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2$
"lstm/lstm_1/while/lstm_cell/mul_10?
 lstm/lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2"
 lstm/lstm_1/while/Tile/multiples?
lstm/lstm_1/while/TileTile>lstm/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0)lstm/lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm/lstm_1/while/Tile?
lstm/lstm_1/while/SelectV2SelectV2lstm/lstm_1/while/Tile:output:0&lstm/lstm_1/while/lstm_cell/mul_10:z:0lstm_lstm_1_while_placeholder_2*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/while/SelectV2?
"lstm/lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm/lstm_1/while/Tile_1/multiples?
lstm/lstm_1/while/Tile_1Tile>lstm/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0+lstm/lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm/lstm_1/while/Tile_1?
"lstm/lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm/lstm_1/while/Tile_2/multiples?
lstm/lstm_1/while/Tile_2Tile>lstm/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0+lstm/lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm/lstm_1/while/Tile_2?
lstm/lstm_1/while/SelectV2_1SelectV2!lstm/lstm_1/while/Tile_1:output:0&lstm/lstm_1/while/lstm_cell/mul_10:z:0lstm_lstm_1_while_placeholder_3*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/while/SelectV2_1?
lstm/lstm_1/while/SelectV2_2SelectV2!lstm/lstm_1/while/Tile_2:output:0%lstm/lstm_1/while/lstm_cell/add_3:z:0lstm_lstm_1_while_placeholder_4*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/while/SelectV2_2?
6lstm/lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_lstm_1_while_placeholder_1lstm_lstm_1_while_placeholder#lstm/lstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype028
6lstm/lstm_1/while/TensorArrayV2Write/TensorListSetItemt
lstm/lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_1/while/add/y?
lstm/lstm_1/while/addAddV2lstm_lstm_1_while_placeholder lstm/lstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/lstm_1/while/addx
lstm/lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_1/while/add_1/y?
lstm/lstm_1/while/add_1AddV20lstm_lstm_1_while_lstm_lstm_1_while_loop_counter"lstm/lstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/lstm_1/while/add_1?
lstm/lstm_1/while/IdentityIdentitylstm/lstm_1/while/add_1:z:0^lstm/lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm/lstm_1/while/Identity?
lstm/lstm_1/while/Identity_1Identity6lstm_lstm_1_while_lstm_lstm_1_while_maximum_iterations^lstm/lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm/lstm_1/while/Identity_1?
lstm/lstm_1/while/Identity_2Identitylstm/lstm_1/while/add:z:0^lstm/lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm/lstm_1/while/Identity_2?
lstm/lstm_1/while/Identity_3IdentityFlstm/lstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm/lstm_1/while/Identity_3?
lstm/lstm_1/while/Identity_4Identity#lstm/lstm_1/while/SelectV2:output:0^lstm/lstm_1/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/while/Identity_4?
lstm/lstm_1/while/Identity_5Identity%lstm/lstm_1/while/SelectV2_1:output:0^lstm/lstm_1/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/while/Identity_5?
lstm/lstm_1/while/Identity_6Identity%lstm/lstm_1/while/SelectV2_2:output:0^lstm/lstm_1/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/while/Identity_6?
lstm/lstm_1/while/NoOpNoOp+^lstm/lstm_1/while/lstm_cell/ReadVariableOp-^lstm/lstm_1/while/lstm_cell/ReadVariableOp_1-^lstm/lstm_1/while/lstm_cell/ReadVariableOp_2-^lstm/lstm_1/while/lstm_cell/ReadVariableOp_31^lstm/lstm_1/while/lstm_cell/split/ReadVariableOp3^lstm/lstm_1/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm/lstm_1/while/NoOp"A
lstm_lstm_1_while_identity#lstm/lstm_1/while/Identity:output:0"E
lstm_lstm_1_while_identity_1%lstm/lstm_1/while/Identity_1:output:0"E
lstm_lstm_1_while_identity_2%lstm/lstm_1/while/Identity_2:output:0"E
lstm_lstm_1_while_identity_3%lstm/lstm_1/while/Identity_3:output:0"E
lstm_lstm_1_while_identity_4%lstm/lstm_1/while/Identity_4:output:0"E
lstm_lstm_1_while_identity_5%lstm/lstm_1/while/Identity_5:output:0"E
lstm_lstm_1_while_identity_6%lstm/lstm_1/while/Identity_6:output:0"l
3lstm_lstm_1_while_lstm_cell_readvariableop_resource5lstm_lstm_1_while_lstm_cell_readvariableop_resource_0"|
;lstm_lstm_1_while_lstm_cell_split_1_readvariableop_resource=lstm_lstm_1_while_lstm_cell_split_1_readvariableop_resource_0"x
9lstm_lstm_1_while_lstm_cell_split_readvariableop_resource;lstm_lstm_1_while_lstm_cell_split_readvariableop_resource_0"`
-lstm_lstm_1_while_lstm_lstm_1_strided_slice_1/lstm_lstm_1_while_lstm_lstm_1_strided_slice_1_0"?
mlstm_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_lstm_1_tensorarrayunstack_1_tensorlistfromtensorolstm_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
ilstm_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_1_tensorarrayunstack_tensorlistfromtensorklstm_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :??????????:??????????:??????????: : : : : : 2X
*lstm/lstm_1/while/lstm_cell/ReadVariableOp*lstm/lstm_1/while/lstm_cell/ReadVariableOp2\
,lstm/lstm_1/while/lstm_cell/ReadVariableOp_1,lstm/lstm_1/while/lstm_cell/ReadVariableOp_12\
,lstm/lstm_1/while/lstm_cell/ReadVariableOp_2,lstm/lstm_1/while/lstm_cell/ReadVariableOp_22\
,lstm/lstm_1/while/lstm_cell/ReadVariableOp_3,lstm/lstm_1/while/lstm_cell/ReadVariableOp_32d
0lstm/lstm_1/while/lstm_cell/split/ReadVariableOp0lstm/lstm_1/while/lstm_cell/split/ReadVariableOp2h
2lstm/lstm_1/while/lstm_cell/split_1/ReadVariableOp2lstm/lstm_1/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?	
?
$__inference_lstm_layer_call_fn_18652
input_1
unknown:	E?
	unknown_0:	?
	unknown_1:
??
	unknown_2:
??
	unknown_3:	?
	unknown_4:	?
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_186352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????E
!
_user_specified_name	input_1
?%
?
while_body_17612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_17636_0:	E?&
while_lstm_cell_17638_0:	?+
while_lstm_cell_17640_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_17636:	E?$
while_lstm_cell_17638:	?)
while_lstm_cell_17640:
????'while/lstm_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_17636_0while_lstm_cell_17638_0while_lstm_cell_17640_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_175982)
'while/lstm_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_17636while_lstm_cell_17636_0"0
while_lstm_cell_17638while_lstm_cell_17638_0"0
while_lstm_cell_17640while_lstm_cell_17640_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
while_cond_18427
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_18427___redundant_placeholder03
/while_while_cond_18427___redundant_placeholder13
/while_while_cond_18427___redundant_placeholder23
/while_while_cond_18427___redundant_placeholder33
/while_while_cond_18427___redundant_placeholder4
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :??????????:??????????:??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
??
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_20994

inputs
mask
:
'lstm_cell_split_readvariableop_resource:	E?8
)lstm_cell_split_1_readvariableop_resource:	?5
!lstm_cell_readvariableop_resource:
??
identity

identity_1

identity_2??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????E2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim{

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:?????????2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:?????????2
transpose_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/ones_likex
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_1?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_3x
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	E?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time?
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2_2/element_shape?
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2?
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensorn

zeros_like	ZerosLikelstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *
bodyR
while_body_20840*
condR
while_cond_20839*c
output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityn

Identity_1Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1n

Identity_2Identitywhile:output:6^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????E:?????????: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????E
 
_user_specified_nameinputs:MI
'
_output_shapes
:?????????

_user_specified_namemask
?	
?
$__inference_lstm_layer_call_fn_20060
x
unknown:	E?
	unknown_0:	?
	unknown_1:
??
	unknown_2:
??
	unknown_3:	?
	unknown_4:	?
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_186352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????E

_user_specified_namex
?
?
while_cond_20512
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_20512___redundant_placeholder03
/while_while_cond_20512___redundant_placeholder13
/while_while_cond_20512___redundant_placeholder23
/while_while_cond_20512___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_17945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_17945___redundant_placeholder03
/while_while_cond_17945___redundant_placeholder13
/while_while_cond_17945___redundant_placeholder23
/while_while_cond_17945___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
ƙ
?

while_body_18428
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	E?@
1while_lstm_cell_split_1_readvariableop_resource_0:	?=
)while_lstm_cell_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	E?>
/while_lstm_cell_split_1_readvariableop_resource:	?;
'while_lstm_cell_readvariableop_resource:
????while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem?
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape?
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/ones_like/Const?
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/ones_like?
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape?
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell/ones_like_1/Const?
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/ones_like_1?
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul?
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_1?
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_2?
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_3?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	E?*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/mul_4Mulwhile_placeholder_3$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_4?
while/lstm_cell/mul_5Mulwhile_placeholder_3$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
while/lstm_cell/mul_6Mulwhile_placeholder_3$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_6?
while/lstm_cell/mul_7Mulwhile_placeholder_3$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_7?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_4*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_8?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2?
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh?
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_9?
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_3?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh_1?
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_10}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2

while/Tile?
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell/mul_10:z:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/SelectV2?
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_1?
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_2?
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell/mul_10:z:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/SelectV2_1?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell/add_3:z:0while_placeholder_4*
T0*(
_output_shapes
:??????????2
while/SelectV2_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?
while/Identity_6Identitywhile/SelectV2_2:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_6?

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :??????????:??????????:??????????: : : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?

?
while_cond_21184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_21184___redundant_placeholder03
/while_while_cond_21184___redundant_placeholder13
/while_while_cond_21184___redundant_placeholder23
/while_while_cond_21184___redundant_placeholder33
/while_while_cond_21184___redundant_placeholder4
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :??????????:??????????:??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?L
?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_17598

inputs

states
states_10
split_readvariableop_resource:	E?.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
	ones_like\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:?????????E2
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????E2
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????E2
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????E2
mul_3d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	E?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3f
mul_4Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_4f
mul_5Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_5f
mul_6Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_6f
mul_7Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10f
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????E:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_21521

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_1_layer_call_fn_21465

inputs
mask

unknown:	E?
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsmaskunknown	unknown_0	unknown_1*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_191232
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????E:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????E
 
_user_specified_nameinputs:MI
'
_output_shapes
:?????????

_user_specified_namemask
?M
?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_21613

inputs
states_0
states_10
split_readvariableop_resource:	E?.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
	ones_like^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:?????????E2
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????E2
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????E2
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????E2
mul_3d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	E?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3h
mul_4Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_4h
mul_5Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_5h
mul_6Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_6h
mul_7Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10f
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????E:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_20713
inputs_0:
'lstm_cell_split_readvariableop_resource:	E?8
)lstm_cell_split_1_readvariableop_resource:	?5
!lstm_cell_readvariableop_resource:
??
identity

identity_1

identity_2??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????E2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_3/Mul_1x
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?22
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_3x
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	E?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_20513*
condR
while_cond_20512*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityn

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1n

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????E: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????E
"
_user_specified_name
inputs/0
?
?
while_cond_17611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_17611___redundant_placeholder03
/while_while_cond_17611___redundant_placeholder13
/while_while_cond_17611___redundant_placeholder23
/while_while_cond_17611___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
)__inference_lstm_cell_layer_call_fn_21793

inputs
states_0
states_1
unknown:	E?
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_178642
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????E:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?	
?
$__inference_lstm_layer_call_fn_20079
x
unknown:	E?
	unknown_0:	?
	unknown_1:
??
	unknown_2:
??
	unknown_3:	?
	unknown_4:	?
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_191872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????E

_user_specified_namex
?
?
&__inference_lstm_1_layer_call_fn_21449

inputs
mask

unknown:	E?
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsmaskunknown	unknown_0	unknown_1*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_185822
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????E:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????E
 
_user_specified_nameinputs:MI
'
_output_shapes
:?????????

_user_specified_namemask
??
?

while_body_21185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	E?@
1while_lstm_cell_split_1_readvariableop_resource_0:	?=
)while_lstm_cell_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	E?>
/while_lstm_cell_split_1_readvariableop_resource:	?;
'while_lstm_cell_readvariableop_resource:
????while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem?
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape?
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/ones_like/Const?
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/ones_like?
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/lstm_cell/dropout/Const?
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout/Mul?
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/Shape?
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???26
4while/lstm_cell/dropout/random_uniform/RandomUniform?
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2(
&while/lstm_cell/dropout/GreaterEqual/y?
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2&
$while/lstm_cell/dropout/GreaterEqual?
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
while/lstm_cell/dropout/Cast?
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout/Mul_1?
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_1/Const?
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout_1/Mul?
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/Shape?
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_1/random_uniform/RandomUniform?
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_1/GreaterEqual/y?
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2(
&while/lstm_cell/dropout_1/GreaterEqual?
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2 
while/lstm_cell/dropout_1/Cast?
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????E2!
while/lstm_cell/dropout_1/Mul_1?
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_2/Const?
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout_2/Mul?
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/Shape?
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_2/random_uniform/RandomUniform?
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_2/GreaterEqual/y?
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2(
&while/lstm_cell/dropout_2/GreaterEqual?
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2 
while/lstm_cell/dropout_2/Cast?
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????E2!
while/lstm_cell/dropout_2/Mul_1?
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_3/Const?
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout_3/Mul?
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/Shape?
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_3/random_uniform/RandomUniform?
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_3/GreaterEqual/y?
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2(
&while/lstm_cell/dropout_3/GreaterEqual?
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2 
while/lstm_cell/dropout_3/Cast?
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????E2!
while/lstm_cell/dropout_3/Mul_1?
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape?
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell/ones_like_1/Const?
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/ones_like_1?
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_4/Const?
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_4/Mul?
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_4/Shape?
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_4/random_uniform/RandomUniform?
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_4/GreaterEqual/y?
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_4/GreaterEqual?
while/lstm_cell/dropout_4/CastCast*while/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_4/Cast?
while/lstm_cell/dropout_4/Mul_1Mul!while/lstm_cell/dropout_4/Mul:z:0"while/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_4/Mul_1?
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_5/Const?
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_5/Mul?
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_5/Shape?
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ӯ28
6while/lstm_cell/dropout_5/random_uniform/RandomUniform?
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_5/GreaterEqual/y?
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_5/GreaterEqual?
while/lstm_cell/dropout_5/CastCast*while/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_5/Cast?
while/lstm_cell/dropout_5/Mul_1Mul!while/lstm_cell/dropout_5/Mul:z:0"while/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_5/Mul_1?
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_6/Const?
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_6/Mul?
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_6/Shape?
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_6/random_uniform/RandomUniform?
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_6/GreaterEqual/y?
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_6/GreaterEqual?
while/lstm_cell/dropout_6/CastCast*while/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_6/Cast?
while/lstm_cell/dropout_6/Mul_1Mul!while/lstm_cell/dropout_6/Mul:z:0"while/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_6/Mul_1?
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_7/Const?
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_7/Mul?
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_7/Shape?
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_7/random_uniform/RandomUniform?
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_7/GreaterEqual/y?
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_7/GreaterEqual?
while/lstm_cell/dropout_7/CastCast*while/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_7/Cast?
while/lstm_cell/dropout_7/Mul_1Mul!while/lstm_cell/dropout_7/Mul:z:0"while/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_7/Mul_1?
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul?
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_1?
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_2?
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_3?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	E?*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/mul_4Mulwhile_placeholder_3#while/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_4?
while/lstm_cell/mul_5Mulwhile_placeholder_3#while/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
while/lstm_cell/mul_6Mulwhile_placeholder_3#while/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_6?
while/lstm_cell/mul_7Mulwhile_placeholder_3#while/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_7?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_4*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_8?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2?
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh?
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_9?
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_3?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh_1?
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_10}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2

while/Tile?
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell/mul_10:z:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/SelectV2?
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_1?
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_2?
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell/mul_10:z:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/SelectV2_1?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell/add_3:z:0while_placeholder_4*
T0*(
_output_shapes
:??????????2
while/SelectV2_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?
while/Identity_6Identitywhile/SelectV2_2:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_6?

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :??????????:??????????:??????????: : : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_21476

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:??????????2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?	
while_body_20513
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	E?@
1while_lstm_cell_split_1_readvariableop_resource_0:	?=
)while_lstm_cell_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	E?>
/while_lstm_cell_split_1_readvariableop_resource:	?;
'while_lstm_cell_readvariableop_resource:
????while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape?
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/ones_like/Const?
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/ones_like?
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/lstm_cell/dropout/Const?
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout/Mul?
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/Shape?
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???26
4while/lstm_cell/dropout/random_uniform/RandomUniform?
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2(
&while/lstm_cell/dropout/GreaterEqual/y?
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2&
$while/lstm_cell/dropout/GreaterEqual?
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
while/lstm_cell/dropout/Cast?
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout/Mul_1?
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_1/Const?
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout_1/Mul?
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/Shape?
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_1/random_uniform/RandomUniform?
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_1/GreaterEqual/y?
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2(
&while/lstm_cell/dropout_1/GreaterEqual?
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2 
while/lstm_cell/dropout_1/Cast?
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????E2!
while/lstm_cell/dropout_1/Mul_1?
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_2/Const?
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout_2/Mul?
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/Shape?
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2ԚF28
6while/lstm_cell/dropout_2/random_uniform/RandomUniform?
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_2/GreaterEqual/y?
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2(
&while/lstm_cell/dropout_2/GreaterEqual?
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2 
while/lstm_cell/dropout_2/Cast?
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????E2!
while/lstm_cell/dropout_2/Mul_1?
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_3/Const?
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout_3/Mul?
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/Shape?
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2??/28
6while/lstm_cell/dropout_3/random_uniform/RandomUniform?
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_3/GreaterEqual/y?
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2(
&while/lstm_cell/dropout_3/GreaterEqual?
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2 
while/lstm_cell/dropout_3/Cast?
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????E2!
while/lstm_cell/dropout_3/Mul_1?
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape?
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell/ones_like_1/Const?
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/ones_like_1?
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_4/Const?
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_4/Mul?
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_4/Shape?
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??28
6while/lstm_cell/dropout_4/random_uniform/RandomUniform?
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_4/GreaterEqual/y?
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_4/GreaterEqual?
while/lstm_cell/dropout_4/CastCast*while/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_4/Cast?
while/lstm_cell/dropout_4/Mul_1Mul!while/lstm_cell/dropout_4/Mul:z:0"while/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_4/Mul_1?
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_5/Const?
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_5/Mul?
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_5/Shape?
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_5/random_uniform/RandomUniform?
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_5/GreaterEqual/y?
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_5/GreaterEqual?
while/lstm_cell/dropout_5/CastCast*while/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_5/Cast?
while/lstm_cell/dropout_5/Mul_1Mul!while/lstm_cell/dropout_5/Mul:z:0"while/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_5/Mul_1?
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_6/Const?
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_6/Mul?
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_6/Shape?
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_6/random_uniform/RandomUniform?
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_6/GreaterEqual/y?
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_6/GreaterEqual?
while/lstm_cell/dropout_6/CastCast*while/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_6/Cast?
while/lstm_cell/dropout_6/Mul_1Mul!while/lstm_cell/dropout_6/Mul:z:0"while/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_6/Mul_1?
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_7/Const?
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_7/Mul?
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_7/Shape?
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ͮ28
6while/lstm_cell/dropout_7/random_uniform/RandomUniform?
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_7/GreaterEqual/y?
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_7/GreaterEqual?
while/lstm_cell/dropout_7/CastCast*while/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_7/Cast?
while/lstm_cell/dropout_7/Mul_1Mul!while/lstm_cell/dropout_7/Mul:z:0"while/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_7/Mul_1?
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul?
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_1?
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_2?
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_3?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	E?*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/mul_4Mulwhile_placeholder_2#while/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_4?
while/lstm_cell/mul_5Mulwhile_placeholder_2#while/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
while/lstm_cell/mul_6Mulwhile_placeholder_2#while/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_6?
while/lstm_cell/mul_7Mulwhile_placeholder_2#while/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_7?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_8?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2?
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh?
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_9?
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_3?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh_1?
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
lstm_1_while_cond_19801*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4,
(lstm_1_while_less_lstm_1_strided_slice_1A
=lstm_1_while_lstm_1_while_cond_19801___redundant_placeholder0A
=lstm_1_while_lstm_1_while_cond_19801___redundant_placeholder1A
=lstm_1_while_lstm_1_while_cond_19801___redundant_placeholder2A
=lstm_1_while_lstm_1_while_cond_19801___redundant_placeholder3A
=lstm_1_while_lstm_1_while_cond_19801___redundant_placeholder4
lstm_1_while_identity
?
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: 2
lstm_1/while/Lessr
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_1/while/Identity"7
lstm_1_while_identitylstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :??????????:??????????:??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
??
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_21403

inputs
mask
:
'lstm_cell_split_readvariableop_resource:	E?8
)lstm_cell_split_1_readvariableop_resource:	?5
!lstm_cell_readvariableop_resource:
??
identity

identity_1

identity_2??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????E2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim{

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:?????????2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:?????????2
transpose_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_3/Mul_1x
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??m22
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ܬ22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??922
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ᠱ22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_3x
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	E?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time?
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2_2/element_shape?
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2?
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensorn

zeros_like	ZerosLikelstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *
bodyR
while_body_21185*
condR
while_cond_21184*c
output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityn

Identity_1Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1n

Identity_2Identitywhile:output:6^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????E:?????????: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????E
 
_user_specified_nameinputs:MI
'
_output_shapes
:?????????

_user_specified_namemask
??
?
?__inference_lstm_layer_call_and_return_conditional_losses_20041
xA
.lstm_1_lstm_cell_split_readvariableop_resource:	E??
0lstm_1_lstm_cell_split_1_readvariableop_resource:	?<
(lstm_1_lstm_cell_readvariableop_resource:
??8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?lstm_1/lstm_cell/ReadVariableOp?!lstm_1/lstm_cell/ReadVariableOp_1?!lstm_1/lstm_cell/ReadVariableOp_2?!lstm_1/lstm_cell/ReadVariableOp_3?%lstm_1/lstm_cell/split/ReadVariableOp?'lstm_1/lstm_cell/split_1/ReadVariableOp?lstm_1/whileO
x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
x_1?
NotEqualNotEqualxx_1:output:0*
T0*+
_output_shapes
:?????????E*
incompatible_shape_error( 2

NotEqualp
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Any/reduction_indicesh
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*'
_output_shapes
:?????????2
AnyM
lstm_1/ShapeShapex*
T0*
_output_shapes
:2
lstm_1/Shape?
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice/stack?
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_1?
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_2?
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slicek
lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros/mul/y?
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/mulm
lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros/Less/y?
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/Lessq
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros/packed/1?
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros/packedm
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros/Const?
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/zeroso
lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros_1/mul/y?
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/mulq
lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros_1/Less/y?
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/Lessu
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros_1/packed/1?
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros_1/packedq
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros_1/Const?
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/zeros_1?
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose/perm?
lstm_1/transpose	Transposexlstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????E2
lstm_1/transposed
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:2
lstm_1/Shape_1?
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_1/stack?
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_1?
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_2?
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slice_1y
lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_1/ExpandDims/dim?
lstm_1/ExpandDims
ExpandDimsAny:output:0lstm_1/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:?????????2
lstm_1/ExpandDims?
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_1/perm?
lstm_1/transpose_1	Transposelstm_1/ExpandDims:output:0 lstm_1/transpose_1/perm:output:0*
T0
*+
_output_shapes
:?????????2
lstm_1/transpose_1?
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_1/TensorArrayV2/element_shape?
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2?
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   2>
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_1/TensorArrayUnstack/TensorListFromTensor?
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_2/stack?
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_1?
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_2?
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
lstm_1/strided_slice_2?
 lstm_1/lstm_cell/ones_like/ShapeShapelstm_1/strided_slice_2:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell/ones_like/Shape?
 lstm_1/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 lstm_1/lstm_cell/ones_like/Const?
lstm_1/lstm_cell/ones_likeFill)lstm_1/lstm_cell/ones_like/Shape:output:0)lstm_1/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/ones_like?
lstm_1/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm_1/lstm_cell/dropout/Const?
lstm_1/lstm_cell/dropout/MulMul#lstm_1/lstm_cell/ones_like:output:0'lstm_1/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/dropout/Mul?
lstm_1/lstm_cell/dropout/ShapeShape#lstm_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2 
lstm_1/lstm_cell/dropout/Shape?
5lstm_1/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform'lstm_1/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???27
5lstm_1/lstm_cell/dropout/random_uniform/RandomUniform?
'lstm_1/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'lstm_1/lstm_cell/dropout/GreaterEqual/y?
%lstm_1/lstm_cell/dropout/GreaterEqualGreaterEqual>lstm_1/lstm_cell/dropout/random_uniform/RandomUniform:output:00lstm_1/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2'
%lstm_1/lstm_cell/dropout/GreaterEqual?
lstm_1/lstm_cell/dropout/CastCast)lstm_1/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/dropout/Cast?
lstm_1/lstm_cell/dropout/Mul_1Mul lstm_1/lstm_cell/dropout/Mul:z:0!lstm_1/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????E2 
lstm_1/lstm_cell/dropout/Mul_1?
 lstm_1/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 lstm_1/lstm_cell/dropout_1/Const?
lstm_1/lstm_cell/dropout_1/MulMul#lstm_1/lstm_cell/ones_like:output:0)lstm_1/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????E2 
lstm_1/lstm_cell/dropout_1/Mul?
 lstm_1/lstm_cell/dropout_1/ShapeShape#lstm_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell/dropout_1/Shape?
7lstm_1/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2?Ԃ29
7lstm_1/lstm_cell/dropout_1/random_uniform/RandomUniform?
)lstm_1/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)lstm_1/lstm_cell/dropout_1/GreaterEqual/y?
'lstm_1/lstm_cell/dropout_1/GreaterEqualGreaterEqual@lstm_1/lstm_cell/dropout_1/random_uniform/RandomUniform:output:02lstm_1/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2)
'lstm_1/lstm_cell/dropout_1/GreaterEqual?
lstm_1/lstm_cell/dropout_1/CastCast+lstm_1/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2!
lstm_1/lstm_cell/dropout_1/Cast?
 lstm_1/lstm_cell/dropout_1/Mul_1Mul"lstm_1/lstm_cell/dropout_1/Mul:z:0#lstm_1/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????E2"
 lstm_1/lstm_cell/dropout_1/Mul_1?
 lstm_1/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 lstm_1/lstm_cell/dropout_2/Const?
lstm_1/lstm_cell/dropout_2/MulMul#lstm_1/lstm_cell/ones_like:output:0)lstm_1/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????E2 
lstm_1/lstm_cell/dropout_2/Mul?
 lstm_1/lstm_cell/dropout_2/ShapeShape#lstm_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell/dropout_2/Shape?
7lstm_1/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2?ͼ29
7lstm_1/lstm_cell/dropout_2/random_uniform/RandomUniform?
)lstm_1/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)lstm_1/lstm_cell/dropout_2/GreaterEqual/y?
'lstm_1/lstm_cell/dropout_2/GreaterEqualGreaterEqual@lstm_1/lstm_cell/dropout_2/random_uniform/RandomUniform:output:02lstm_1/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2)
'lstm_1/lstm_cell/dropout_2/GreaterEqual?
lstm_1/lstm_cell/dropout_2/CastCast+lstm_1/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2!
lstm_1/lstm_cell/dropout_2/Cast?
 lstm_1/lstm_cell/dropout_2/Mul_1Mul"lstm_1/lstm_cell/dropout_2/Mul:z:0#lstm_1/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????E2"
 lstm_1/lstm_cell/dropout_2/Mul_1?
 lstm_1/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 lstm_1/lstm_cell/dropout_3/Const?
lstm_1/lstm_cell/dropout_3/MulMul#lstm_1/lstm_cell/ones_like:output:0)lstm_1/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????E2 
lstm_1/lstm_cell/dropout_3/Mul?
 lstm_1/lstm_cell/dropout_3/ShapeShape#lstm_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell/dropout_3/Shape?
7lstm_1/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???29
7lstm_1/lstm_cell/dropout_3/random_uniform/RandomUniform?
)lstm_1/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)lstm_1/lstm_cell/dropout_3/GreaterEqual/y?
'lstm_1/lstm_cell/dropout_3/GreaterEqualGreaterEqual@lstm_1/lstm_cell/dropout_3/random_uniform/RandomUniform:output:02lstm_1/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2)
'lstm_1/lstm_cell/dropout_3/GreaterEqual?
lstm_1/lstm_cell/dropout_3/CastCast+lstm_1/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2!
lstm_1/lstm_cell/dropout_3/Cast?
 lstm_1/lstm_cell/dropout_3/Mul_1Mul"lstm_1/lstm_cell/dropout_3/Mul:z:0#lstm_1/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????E2"
 lstm_1/lstm_cell/dropout_3/Mul_1?
"lstm_1/lstm_cell/ones_like_1/ShapeShapelstm_1/zeros:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell/ones_like_1/Shape?
"lstm_1/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"lstm_1/lstm_cell/ones_like_1/Const?
lstm_1/lstm_cell/ones_like_1Fill+lstm_1/lstm_cell/ones_like_1/Shape:output:0+lstm_1/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/ones_like_1?
 lstm_1/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 lstm_1/lstm_cell/dropout_4/Const?
lstm_1/lstm_cell/dropout_4/MulMul%lstm_1/lstm_cell/ones_like_1:output:0)lstm_1/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2 
lstm_1/lstm_cell/dropout_4/Mul?
 lstm_1/lstm_cell/dropout_4/ShapeShape%lstm_1/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell/dropout_4/Shape?
7lstm_1/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??=29
7lstm_1/lstm_cell/dropout_4/random_uniform/RandomUniform?
)lstm_1/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)lstm_1/lstm_cell/dropout_4/GreaterEqual/y?
'lstm_1/lstm_cell/dropout_4/GreaterEqualGreaterEqual@lstm_1/lstm_cell/dropout_4/random_uniform/RandomUniform:output:02lstm_1/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'lstm_1/lstm_cell/dropout_4/GreaterEqual?
lstm_1/lstm_cell/dropout_4/CastCast+lstm_1/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
lstm_1/lstm_cell/dropout_4/Cast?
 lstm_1/lstm_cell/dropout_4/Mul_1Mul"lstm_1/lstm_cell/dropout_4/Mul:z:0#lstm_1/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 lstm_1/lstm_cell/dropout_4/Mul_1?
 lstm_1/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 lstm_1/lstm_cell/dropout_5/Const?
lstm_1/lstm_cell/dropout_5/MulMul%lstm_1/lstm_cell/ones_like_1:output:0)lstm_1/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2 
lstm_1/lstm_cell/dropout_5/Mul?
 lstm_1/lstm_cell/dropout_5/ShapeShape%lstm_1/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell/dropout_5/Shape?
7lstm_1/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?29
7lstm_1/lstm_cell/dropout_5/random_uniform/RandomUniform?
)lstm_1/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)lstm_1/lstm_cell/dropout_5/GreaterEqual/y?
'lstm_1/lstm_cell/dropout_5/GreaterEqualGreaterEqual@lstm_1/lstm_cell/dropout_5/random_uniform/RandomUniform:output:02lstm_1/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'lstm_1/lstm_cell/dropout_5/GreaterEqual?
lstm_1/lstm_cell/dropout_5/CastCast+lstm_1/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
lstm_1/lstm_cell/dropout_5/Cast?
 lstm_1/lstm_cell/dropout_5/Mul_1Mul"lstm_1/lstm_cell/dropout_5/Mul:z:0#lstm_1/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 lstm_1/lstm_cell/dropout_5/Mul_1?
 lstm_1/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 lstm_1/lstm_cell/dropout_6/Const?
lstm_1/lstm_cell/dropout_6/MulMul%lstm_1/lstm_cell/ones_like_1:output:0)lstm_1/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2 
lstm_1/lstm_cell/dropout_6/Mul?
 lstm_1/lstm_cell/dropout_6/ShapeShape%lstm_1/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell/dropout_6/Shape?
7lstm_1/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???29
7lstm_1/lstm_cell/dropout_6/random_uniform/RandomUniform?
)lstm_1/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)lstm_1/lstm_cell/dropout_6/GreaterEqual/y?
'lstm_1/lstm_cell/dropout_6/GreaterEqualGreaterEqual@lstm_1/lstm_cell/dropout_6/random_uniform/RandomUniform:output:02lstm_1/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'lstm_1/lstm_cell/dropout_6/GreaterEqual?
lstm_1/lstm_cell/dropout_6/CastCast+lstm_1/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
lstm_1/lstm_cell/dropout_6/Cast?
 lstm_1/lstm_cell/dropout_6/Mul_1Mul"lstm_1/lstm_cell/dropout_6/Mul:z:0#lstm_1/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 lstm_1/lstm_cell/dropout_6/Mul_1?
 lstm_1/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 lstm_1/lstm_cell/dropout_7/Const?
lstm_1/lstm_cell/dropout_7/MulMul%lstm_1/lstm_cell/ones_like_1:output:0)lstm_1/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2 
lstm_1/lstm_cell/dropout_7/Mul?
 lstm_1/lstm_cell/dropout_7/ShapeShape%lstm_1/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell/dropout_7/Shape?
7lstm_1/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??629
7lstm_1/lstm_cell/dropout_7/random_uniform/RandomUniform?
)lstm_1/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)lstm_1/lstm_cell/dropout_7/GreaterEqual/y?
'lstm_1/lstm_cell/dropout_7/GreaterEqualGreaterEqual@lstm_1/lstm_cell/dropout_7/random_uniform/RandomUniform:output:02lstm_1/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'lstm_1/lstm_cell/dropout_7/GreaterEqual?
lstm_1/lstm_cell/dropout_7/CastCast+lstm_1/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
lstm_1/lstm_cell/dropout_7/Cast?
 lstm_1/lstm_cell/dropout_7/Mul_1Mul"lstm_1/lstm_cell/dropout_7/Mul:z:0#lstm_1/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 lstm_1/lstm_cell/dropout_7/Mul_1?
lstm_1/lstm_cell/mulMullstm_1/strided_slice_2:output:0"lstm_1/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/mul?
lstm_1/lstm_cell/mul_1Mullstm_1/strided_slice_2:output:0$lstm_1/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/mul_1?
lstm_1/lstm_cell/mul_2Mullstm_1/strided_slice_2:output:0$lstm_1/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/mul_2?
lstm_1/lstm_cell/mul_3Mullstm_1/strided_slice_2:output:0$lstm_1/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/mul_3?
 lstm_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_1/lstm_cell/split/split_dim?
%lstm_1/lstm_cell/split/ReadVariableOpReadVariableOp.lstm_1_lstm_cell_split_readvariableop_resource*
_output_shapes
:	E?*
dtype02'
%lstm_1/lstm_cell/split/ReadVariableOp?
lstm_1/lstm_cell/splitSplit)lstm_1/lstm_cell/split/split_dim:output:0-lstm_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm_1/lstm_cell/split?
lstm_1/lstm_cell/MatMulMatMullstm_1/lstm_cell/mul:z:0lstm_1/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul?
lstm_1/lstm_cell/MatMul_1MatMullstm_1/lstm_cell/mul_1:z:0lstm_1/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_1?
lstm_1/lstm_cell/MatMul_2MatMullstm_1/lstm_cell/mul_2:z:0lstm_1/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_2?
lstm_1/lstm_cell/MatMul_3MatMullstm_1/lstm_cell/mul_3:z:0lstm_1/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_3?
"lstm_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lstm_1/lstm_cell/split_1/split_dim?
'lstm_1/lstm_cell/split_1/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'lstm_1/lstm_cell/split_1/ReadVariableOp?
lstm_1/lstm_cell/split_1Split+lstm_1/lstm_cell/split_1/split_dim:output:0/lstm_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_1/lstm_cell/split_1?
lstm_1/lstm_cell/BiasAddBiasAdd!lstm_1/lstm_cell/MatMul:product:0!lstm_1/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/BiasAdd?
lstm_1/lstm_cell/BiasAdd_1BiasAdd#lstm_1/lstm_cell/MatMul_1:product:0!lstm_1/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/BiasAdd_1?
lstm_1/lstm_cell/BiasAdd_2BiasAdd#lstm_1/lstm_cell/MatMul_2:product:0!lstm_1/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/BiasAdd_2?
lstm_1/lstm_cell/BiasAdd_3BiasAdd#lstm_1/lstm_cell/MatMul_3:product:0!lstm_1/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/BiasAdd_3?
lstm_1/lstm_cell/mul_4Mullstm_1/zeros:output:0$lstm_1/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_4?
lstm_1/lstm_cell/mul_5Mullstm_1/zeros:output:0$lstm_1/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_5?
lstm_1/lstm_cell/mul_6Mullstm_1/zeros:output:0$lstm_1/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_6?
lstm_1/lstm_cell/mul_7Mullstm_1/zeros:output:0$lstm_1/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_7?
lstm_1/lstm_cell/ReadVariableOpReadVariableOp(lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
lstm_1/lstm_cell/ReadVariableOp?
$lstm_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_1/lstm_cell/strided_slice/stack?
&lstm_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&lstm_1/lstm_cell/strided_slice/stack_1?
&lstm_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm_1/lstm_cell/strided_slice/stack_2?
lstm_1/lstm_cell/strided_sliceStridedSlice'lstm_1/lstm_cell/ReadVariableOp:value:0-lstm_1/lstm_cell/strided_slice/stack:output:0/lstm_1/lstm_cell/strided_slice/stack_1:output:0/lstm_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
lstm_1/lstm_cell/strided_slice?
lstm_1/lstm_cell/MatMul_4MatMullstm_1/lstm_cell/mul_4:z:0'lstm_1/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_4?
lstm_1/lstm_cell/addAddV2!lstm_1/lstm_cell/BiasAdd:output:0#lstm_1/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/add?
lstm_1/lstm_cell/SigmoidSigmoidlstm_1/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/Sigmoid?
!lstm_1/lstm_cell/ReadVariableOp_1ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_1/lstm_cell/ReadVariableOp_1?
&lstm_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2(
&lstm_1/lstm_cell/strided_slice_1/stack?
(lstm_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell/strided_slice_1/stack_1?
(lstm_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell/strided_slice_1/stack_2?
 lstm_1/lstm_cell/strided_slice_1StridedSlice)lstm_1/lstm_cell/ReadVariableOp_1:value:0/lstm_1/lstm_cell/strided_slice_1/stack:output:01lstm_1/lstm_cell/strided_slice_1/stack_1:output:01lstm_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 lstm_1/lstm_cell/strided_slice_1?
lstm_1/lstm_cell/MatMul_5MatMullstm_1/lstm_cell/mul_5:z:0)lstm_1/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_5?
lstm_1/lstm_cell/add_1AddV2#lstm_1/lstm_cell/BiasAdd_1:output:0#lstm_1/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/add_1?
lstm_1/lstm_cell/Sigmoid_1Sigmoidlstm_1/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/Sigmoid_1?
lstm_1/lstm_cell/mul_8Mullstm_1/lstm_cell/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_8?
!lstm_1/lstm_cell/ReadVariableOp_2ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_1/lstm_cell/ReadVariableOp_2?
&lstm_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2(
&lstm_1/lstm_cell/strided_slice_2/stack?
(lstm_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell/strided_slice_2/stack_1?
(lstm_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell/strided_slice_2/stack_2?
 lstm_1/lstm_cell/strided_slice_2StridedSlice)lstm_1/lstm_cell/ReadVariableOp_2:value:0/lstm_1/lstm_cell/strided_slice_2/stack:output:01lstm_1/lstm_cell/strided_slice_2/stack_1:output:01lstm_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 lstm_1/lstm_cell/strided_slice_2?
lstm_1/lstm_cell/MatMul_6MatMullstm_1/lstm_cell/mul_6:z:0)lstm_1/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_6?
lstm_1/lstm_cell/add_2AddV2#lstm_1/lstm_cell/BiasAdd_2:output:0#lstm_1/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/add_2?
lstm_1/lstm_cell/TanhTanhlstm_1/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/Tanh?
lstm_1/lstm_cell/mul_9Mullstm_1/lstm_cell/Sigmoid:y:0lstm_1/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_9?
lstm_1/lstm_cell/add_3AddV2lstm_1/lstm_cell/mul_8:z:0lstm_1/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/add_3?
!lstm_1/lstm_cell/ReadVariableOp_3ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_1/lstm_cell/ReadVariableOp_3?
&lstm_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2(
&lstm_1/lstm_cell/strided_slice_3/stack?
(lstm_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_1/lstm_cell/strided_slice_3/stack_1?
(lstm_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell/strided_slice_3/stack_2?
 lstm_1/lstm_cell/strided_slice_3StridedSlice)lstm_1/lstm_cell/ReadVariableOp_3:value:0/lstm_1/lstm_cell/strided_slice_3/stack:output:01lstm_1/lstm_cell/strided_slice_3/stack_1:output:01lstm_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 lstm_1/lstm_cell/strided_slice_3?
lstm_1/lstm_cell/MatMul_7MatMullstm_1/lstm_cell/mul_7:z:0)lstm_1/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_7?
lstm_1/lstm_cell/add_4AddV2#lstm_1/lstm_cell/BiasAdd_3:output:0#lstm_1/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/add_4?
lstm_1/lstm_cell/Sigmoid_2Sigmoidlstm_1/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/Sigmoid_2?
lstm_1/lstm_cell/Tanh_1Tanhlstm_1/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/Tanh_1?
lstm_1/lstm_cell/mul_10Mullstm_1/lstm_cell/Sigmoid_2:y:0lstm_1/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_10?
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$lstm_1/TensorArrayV2_1/element_shape?
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2_1\
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/time?
$lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$lstm_1/TensorArrayV2_2/element_shape?
lstm_1/TensorArrayV2_2TensorListReserve-lstm_1/TensorArrayV2_2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
lstm_1/TensorArrayV2_2?
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
0lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm_1/transpose_1:y:0Glstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type022
0lstm_1/TensorArrayUnstack_1/TensorListFromTensor?
lstm_1/zeros_like	ZerosLikelstm_1/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/zeros_like?
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_1/while/maximum_iterationsx
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/while/loop_counter?
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros_like:y:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0.lstm_1_lstm_cell_split_readvariableop_resource0lstm_1_lstm_cell_split_1_readvariableop_resource(lstm_1_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *#
bodyR
lstm_1_while_body_19802*#
condR
lstm_1_while_cond_19801*c
output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *
parallel_iterations 2
lstm_1/while?
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)lstm_1/TensorArrayV2Stack/TensorListStack?
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_1/strided_slice_3/stack?
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_1/strided_slice_3/stack_1?
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_3/stack_2?
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_1/strided_slice_3?
lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_2/perm?
lstm_1/transpose_2	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_1/transpose_2t
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2lstm_1/strided_slice_3:output:0lstm_1/while:output:5lstm_1/while:output:6concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddq
dense/LeakyRelu	LeakyReludense/BiasAdd:output:0*(
_output_shapes
:??????????2
dense/LeakyRelus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMuldense/LeakyRelu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul{
dropout/dropout/ShapeShapedense/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdds
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^lstm_1/lstm_cell/ReadVariableOp"^lstm_1/lstm_cell/ReadVariableOp_1"^lstm_1/lstm_cell/ReadVariableOp_2"^lstm_1/lstm_cell/ReadVariableOp_3&^lstm_1/lstm_cell/split/ReadVariableOp(^lstm_1/lstm_cell/split_1/ReadVariableOp^lstm_1/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
lstm_1/lstm_cell/ReadVariableOplstm_1/lstm_cell/ReadVariableOp2F
!lstm_1/lstm_cell/ReadVariableOp_1!lstm_1/lstm_cell/ReadVariableOp_12F
!lstm_1/lstm_cell/ReadVariableOp_2!lstm_1/lstm_cell/ReadVariableOp_22F
!lstm_1/lstm_cell/ReadVariableOp_3!lstm_1/lstm_cell/ReadVariableOp_32N
%lstm_1/lstm_cell/split/ReadVariableOp%lstm_1/lstm_cell/split/ReadVariableOp2R
'lstm_1/lstm_cell/split_1/ReadVariableOp'lstm_1/lstm_cell/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:N J
+
_output_shapes
:?????????E

_user_specified_namex
?
?
&__inference_lstm_1_layer_call_fn_21418
inputs_0
unknown:	E?
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_176832
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????E: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????E
"
_user_specified_name
inputs/0
??
?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_21759

inputs
states_0
states_10
split_readvariableop_resource:	E?.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????E2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????E2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????E2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????E2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????E2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????E2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????E2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2?б2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????E2
dropout_3/Mul_1^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/Const?
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_5/Const?
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_6/Const?
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shape?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2Ʈq2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_6/GreaterEqual/y?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/GreaterEqual?
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_6/Cast?
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_7/Const?
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shape?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_7/GreaterEqual/y?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/GreaterEqual?
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_7/Cast?
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
mul_3d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	E?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3g
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_4g
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_5g
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_6g
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10f
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????E:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?	
?
$__inference_lstm_layer_call_fn_19223
input_1
unknown:	E?
	unknown_0:	?
	unknown_1:
??
	unknown_2:
??
	unknown_3:	?
	unknown_4:	?
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_191872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????E
!
_user_specified_name	input_1
?
?
?__inference_lstm_layer_call_and_return_conditional_losses_19187
x
lstm_1_19164:	E?
lstm_1_19166:	? 
lstm_1_19168:
??
dense_19175:
??
dense_19177:	? 
dense_1_19181:	?
dense_1_19183:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?lstm_1/StatefulPartitionedCallO
x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
x_1?
NotEqualNotEqualxx_1:output:0*
T0*+
_output_shapes
:?????????E*
incompatible_shape_error( 2

NotEqualp
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Any/reduction_indicesh
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*'
_output_shapes
:?????????2
Any?
lstm_1/StatefulPartitionedCallStatefulPartitionedCallxAny:output:0lstm_1_19164lstm_1_19166lstm_1_19168*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_191232 
lstm_1/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2'lstm_1/StatefulPartitionedCall:output:0'lstm_1/StatefulPartitionedCall:output:1'lstm_1/StatefulPartitionedCall:output:2concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_19175dense_19177*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_186052
dense/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_186822!
dropout/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_19181dense_1_19183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_186282!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:N J
+
_output_shapes
:?????????E

_user_specified_namex
??
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_18582

inputs
mask
:
'lstm_cell_split_readvariableop_resource:	E?8
)lstm_cell_split_1_readvariableop_resource:	?5
!lstm_cell_readvariableop_resource:
??
identity

identity_1

identity_2??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????E2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim{

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:?????????2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:?????????2
transpose_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/ones_likex
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_1?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_3x
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	E?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time?
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2_2/element_shape?
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2?
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensorn

zeros_like	ZerosLikelstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *
bodyR
while_body_18428*
condR
while_cond_18427*c
output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityn

Identity_1Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1n

Identity_2Identitywhile:output:6^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????E:?????????: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????E
 
_user_specified_nameinputs:MI
'
_output_shapes
:?????????

_user_specified_namemask
?
?
)__inference_lstm_cell_layer_call_fn_21776

inputs
states_0
states_1
unknown:	E?
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_175982
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????E:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?H
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_18017

inputs"
lstm_cell_17933:	E?
lstm_cell_17935:	?#
lstm_cell_17937:
??
identity

identity_1

identity_2??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????E2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_17933lstm_cell_17935lstm_cell_17937*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_178642#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_17933lstm_cell_17935lstm_cell_17937*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_17946*
condR
while_cond_17945*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityn

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1n

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2z
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????E: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????E
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_22006
file_prefix6
"assignvariableop_lstm_dense_kernel:
??1
"assignvariableop_1_lstm_dense_bias:	?9
&assignvariableop_2_lstm_dense_1_kernel:	?2
$assignvariableop_3_lstm_dense_1_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: B
/assignvariableop_9_lstm_lstm_1_lstm_cell_kernel:	E?N
:assignvariableop_10_lstm_lstm_1_lstm_cell_recurrent_kernel:
??=
.assignvariableop_11_lstm_lstm_1_lstm_cell_bias:	?#
assignvariableop_12_total: #
assignvariableop_13_count: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: @
,assignvariableop_16_adam_lstm_dense_kernel_m:
??9
*assignvariableop_17_adam_lstm_dense_bias_m:	?A
.assignvariableop_18_adam_lstm_dense_1_kernel_m:	?:
,assignvariableop_19_adam_lstm_dense_1_bias_m:J
7assignvariableop_20_adam_lstm_lstm_1_lstm_cell_kernel_m:	E?U
Aassignvariableop_21_adam_lstm_lstm_1_lstm_cell_recurrent_kernel_m:
??D
5assignvariableop_22_adam_lstm_lstm_1_lstm_cell_bias_m:	?@
,assignvariableop_23_adam_lstm_dense_kernel_v:
??9
*assignvariableop_24_adam_lstm_dense_bias_v:	?A
.assignvariableop_25_adam_lstm_dense_1_kernel_v:	?:
,assignvariableop_26_adam_lstm_dense_1_bias_v:J
7assignvariableop_27_adam_lstm_lstm_1_lstm_cell_kernel_v:	E?U
Aassignvariableop_28_adam_lstm_lstm_1_lstm_cell_recurrent_kernel_v:
??D
5assignvariableop_29_adam_lstm_lstm_1_lstm_cell_bias_v:	?
identity_31??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_lstm_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_lstm_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_lstm_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_lstm_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp/assignvariableop_9_lstm_lstm_1_lstm_cell_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp:assignvariableop_10_lstm_lstm_1_lstm_cell_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp.assignvariableop_11_lstm_lstm_1_lstm_cell_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp,assignvariableop_16_adam_lstm_dense_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_lstm_dense_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_adam_lstm_dense_1_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_lstm_dense_1_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp7assignvariableop_20_adam_lstm_lstm_1_lstm_cell_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpAassignvariableop_21_adam_lstm_lstm_1_lstm_cell_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_lstm_lstm_1_lstm_cell_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_lstm_dense_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_lstm_dense_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_lstm_dense_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_lstm_dense_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adam_lstm_lstm_1_lstm_cell_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpAassignvariableop_28_adam_lstm_lstm_1_lstm_cell_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp5assignvariableop_29_adam_lstm_lstm_1_lstm_cell_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_299
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_30f
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_31?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
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
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_21495

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?H
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_17683

inputs"
lstm_cell_17599:	E?
lstm_cell_17601:	?#
lstm_cell_17603:
??
identity

identity_1

identity_2??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????E2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_17599lstm_cell_17601lstm_cell_17603*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_175982#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_17599lstm_cell_17601lstm_cell_17603*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_17612*
condR
while_cond_17611*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityn

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1n

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2z
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????E: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????E
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_18616

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ل
?	
while_body_20196
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	E?@
1while_lstm_cell_split_1_readvariableop_resource_0:	?=
)while_lstm_cell_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	E?>
/while_lstm_cell_split_1_readvariableop_resource:	?;
'while_lstm_cell_readvariableop_resource:
????while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape?
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/ones_like/Const?
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/ones_like?
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape?
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell/ones_like_1/Const?
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/ones_like_1?
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul?
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_1?
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_2?
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_3?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	E?*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/mul_4Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_4?
while/lstm_cell/mul_5Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
while/lstm_cell/mul_6Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_6?
while/lstm_cell/mul_7Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_7?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_8?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2?
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh?
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_9?
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_3?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh_1?
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
ƙ
?

while_body_20840
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	E?@
1while_lstm_cell_split_1_readvariableop_resource_0:	?=
)while_lstm_cell_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	E?>
/while_lstm_cell_split_1_readvariableop_resource:	?;
'while_lstm_cell_readvariableop_resource:
????while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem?
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape?
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/ones_like/Const?
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/ones_like?
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape?
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell/ones_like_1/Const?
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/ones_like_1?
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul?
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_1?
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_2?
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_3?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	E?*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/mul_4Mulwhile_placeholder_3$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_4?
while/lstm_cell/mul_5Mulwhile_placeholder_3$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
while/lstm_cell/mul_6Mulwhile_placeholder_3$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_6?
while/lstm_cell/mul_7Mulwhile_placeholder_3$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_7?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_4*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_8?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2?
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh?
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_9?
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_3?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh_1?
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_10}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2

while/Tile?
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell/mul_10:z:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/SelectV2?
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_1?
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_2?
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell/mul_10:z:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/SelectV2_1?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell/add_3:z:0while_placeholder_4*
T0*(
_output_shapes
:??????????2
while/SelectV2_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?
while/Identity_6Identitywhile/SelectV2_2:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_6?

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :??????????:??????????:??????????: : : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
??
?
 __inference__wrapped_model_17473
input_1F
3lstm_lstm_1_lstm_cell_split_readvariableop_resource:	E?D
5lstm_lstm_1_lstm_cell_split_1_readvariableop_resource:	?A
-lstm_lstm_1_lstm_cell_readvariableop_resource:
??=
)lstm_dense_matmul_readvariableop_resource:
??9
*lstm_dense_biasadd_readvariableop_resource:	?>
+lstm_dense_1_matmul_readvariableop_resource:	?:
,lstm_dense_1_biasadd_readvariableop_resource:
identity??!lstm/dense/BiasAdd/ReadVariableOp? lstm/dense/MatMul/ReadVariableOp?#lstm/dense_1/BiasAdd/ReadVariableOp?"lstm/dense_1/MatMul/ReadVariableOp?$lstm/lstm_1/lstm_cell/ReadVariableOp?&lstm/lstm_1/lstm_cell/ReadVariableOp_1?&lstm/lstm_1/lstm_cell/ReadVariableOp_2?&lstm/lstm_1/lstm_cell/ReadVariableOp_3?*lstm/lstm_1/lstm_cell/split/ReadVariableOp?,lstm/lstm_1/lstm_cell/split_1/ReadVariableOp?lstm/lstm_1/whileU
lstm/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm/x?
lstm/NotEqualNotEqualinput_1lstm/x:output:0*
T0*+
_output_shapes
:?????????E*
incompatible_shape_error( 2
lstm/NotEqualz
lstm/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/Any/reduction_indices|
lstm/AnyAnylstm/NotEqual:z:0#lstm/Any/reduction_indices:output:0*'
_output_shapes
:?????????2

lstm/Any]
lstm/lstm_1/ShapeShapeinput_1*
T0*
_output_shapes
:2
lstm/lstm_1/Shape?
lstm/lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
lstm/lstm_1/strided_slice/stack?
!lstm/lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm/lstm_1/strided_slice/stack_1?
!lstm/lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm/lstm_1/strided_slice/stack_2?
lstm/lstm_1/strided_sliceStridedSlicelstm/lstm_1/Shape:output:0(lstm/lstm_1/strided_slice/stack:output:0*lstm/lstm_1/strided_slice/stack_1:output:0*lstm/lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/lstm_1/strided_sliceu
lstm/lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/lstm_1/zeros/mul/y?
lstm/lstm_1/zeros/mulMul"lstm/lstm_1/strided_slice:output:0 lstm/lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/lstm_1/zeros/mulw
lstm/lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/lstm_1/zeros/Less/y?
lstm/lstm_1/zeros/LessLesslstm/lstm_1/zeros/mul:z:0!lstm/lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/lstm_1/zeros/Less{
lstm/lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm/lstm_1/zeros/packed/1?
lstm/lstm_1/zeros/packedPack"lstm/lstm_1/strided_slice:output:0#lstm/lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/lstm_1/zeros/packedw
lstm/lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/lstm_1/zeros/Const?
lstm/lstm_1/zerosFill!lstm/lstm_1/zeros/packed:output:0 lstm/lstm_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/zerosy
lstm/lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/lstm_1/zeros_1/mul/y?
lstm/lstm_1/zeros_1/mulMul"lstm/lstm_1/strided_slice:output:0"lstm/lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/lstm_1/zeros_1/mul{
lstm/lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/lstm_1/zeros_1/Less/y?
lstm/lstm_1/zeros_1/LessLesslstm/lstm_1/zeros_1/mul:z:0#lstm/lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/lstm_1/zeros_1/Less
lstm/lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm/lstm_1/zeros_1/packed/1?
lstm/lstm_1/zeros_1/packedPack"lstm/lstm_1/strided_slice:output:0%lstm/lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/lstm_1/zeros_1/packed{
lstm/lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/lstm_1/zeros_1/Const?
lstm/lstm_1/zeros_1Fill#lstm/lstm_1/zeros_1/packed:output:0"lstm/lstm_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/zeros_1?
lstm/lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/lstm_1/transpose/perm?
lstm/lstm_1/transpose	Transposeinput_1#lstm/lstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????E2
lstm/lstm_1/transposes
lstm/lstm_1/Shape_1Shapelstm/lstm_1/transpose:y:0*
T0*
_output_shapes
:2
lstm/lstm_1/Shape_1?
!lstm/lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!lstm/lstm_1/strided_slice_1/stack?
#lstm/lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#lstm/lstm_1/strided_slice_1/stack_1?
#lstm/lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#lstm/lstm_1/strided_slice_1/stack_2?
lstm/lstm_1/strided_slice_1StridedSlicelstm/lstm_1/Shape_1:output:0*lstm/lstm_1/strided_slice_1/stack:output:0,lstm/lstm_1/strided_slice_1/stack_1:output:0,lstm/lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/lstm_1/strided_slice_1?
lstm/lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm/lstm_1/ExpandDims/dim?
lstm/lstm_1/ExpandDims
ExpandDimslstm/Any:output:0#lstm/lstm_1/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:?????????2
lstm/lstm_1/ExpandDims?
lstm/lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/lstm_1/transpose_1/perm?
lstm/lstm_1/transpose_1	Transposelstm/lstm_1/ExpandDims:output:0%lstm/lstm_1/transpose_1/perm:output:0*
T0
*+
_output_shapes
:?????????2
lstm/lstm_1/transpose_1?
'lstm/lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'lstm/lstm_1/TensorArrayV2/element_shape?
lstm/lstm_1/TensorArrayV2TensorListReserve0lstm/lstm_1/TensorArrayV2/element_shape:output:0$lstm/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/lstm_1/TensorArrayV2?
Alstm/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   2C
Alstm/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
3lstm/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/lstm_1/transpose:y:0Jlstm/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type025
3lstm/lstm_1/TensorArrayUnstack/TensorListFromTensor?
!lstm/lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!lstm/lstm_1/strided_slice_2/stack?
#lstm/lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#lstm/lstm_1/strided_slice_2/stack_1?
#lstm/lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#lstm/lstm_1/strided_slice_2/stack_2?
lstm/lstm_1/strided_slice_2StridedSlicelstm/lstm_1/transpose:y:0*lstm/lstm_1/strided_slice_2/stack:output:0,lstm/lstm_1/strided_slice_2/stack_1:output:0,lstm/lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
lstm/lstm_1/strided_slice_2?
%lstm/lstm_1/lstm_cell/ones_like/ShapeShape$lstm/lstm_1/strided_slice_2:output:0*
T0*
_output_shapes
:2'
%lstm/lstm_1/lstm_cell/ones_like/Shape?
%lstm/lstm_1/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%lstm/lstm_1/lstm_cell/ones_like/Const?
lstm/lstm_1/lstm_cell/ones_likeFill.lstm/lstm_1/lstm_cell/ones_like/Shape:output:0.lstm/lstm_1/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2!
lstm/lstm_1/lstm_cell/ones_like?
'lstm/lstm_1/lstm_cell/ones_like_1/ShapeShapelstm/lstm_1/zeros:output:0*
T0*
_output_shapes
:2)
'lstm/lstm_1/lstm_cell/ones_like_1/Shape?
'lstm/lstm_1/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'lstm/lstm_1/lstm_cell/ones_like_1/Const?
!lstm/lstm_1/lstm_cell/ones_like_1Fill0lstm/lstm_1/lstm_cell/ones_like_1/Shape:output:00lstm/lstm_1/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2#
!lstm/lstm_1/lstm_cell/ones_like_1?
lstm/lstm_1/lstm_cell/mulMul$lstm/lstm_1/strided_slice_2:output:0(lstm/lstm_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm/lstm_1/lstm_cell/mul?
lstm/lstm_1/lstm_cell/mul_1Mul$lstm/lstm_1/strided_slice_2:output:0(lstm/lstm_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm/lstm_1/lstm_cell/mul_1?
lstm/lstm_1/lstm_cell/mul_2Mul$lstm/lstm_1/strided_slice_2:output:0(lstm/lstm_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm/lstm_1/lstm_cell/mul_2?
lstm/lstm_1/lstm_cell/mul_3Mul$lstm/lstm_1/strided_slice_2:output:0(lstm/lstm_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm/lstm_1/lstm_cell/mul_3?
%lstm/lstm_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%lstm/lstm_1/lstm_cell/split/split_dim?
*lstm/lstm_1/lstm_cell/split/ReadVariableOpReadVariableOp3lstm_lstm_1_lstm_cell_split_readvariableop_resource*
_output_shapes
:	E?*
dtype02,
*lstm/lstm_1/lstm_cell/split/ReadVariableOp?
lstm/lstm_1/lstm_cell/splitSplit.lstm/lstm_1/lstm_cell/split/split_dim:output:02lstm/lstm_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm/lstm_1/lstm_cell/split?
lstm/lstm_1/lstm_cell/MatMulMatMullstm/lstm_1/lstm_cell/mul:z:0$lstm/lstm_1/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/MatMul?
lstm/lstm_1/lstm_cell/MatMul_1MatMullstm/lstm_1/lstm_cell/mul_1:z:0$lstm/lstm_1/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm/lstm_1/lstm_cell/MatMul_1?
lstm/lstm_1/lstm_cell/MatMul_2MatMullstm/lstm_1/lstm_cell/mul_2:z:0$lstm/lstm_1/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2 
lstm/lstm_1/lstm_cell/MatMul_2?
lstm/lstm_1/lstm_cell/MatMul_3MatMullstm/lstm_1/lstm_cell/mul_3:z:0$lstm/lstm_1/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm/lstm_1/lstm_cell/MatMul_3?
'lstm/lstm_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'lstm/lstm_1/lstm_cell/split_1/split_dim?
,lstm/lstm_1/lstm_cell/split_1/ReadVariableOpReadVariableOp5lstm_lstm_1_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,lstm/lstm_1/lstm_cell/split_1/ReadVariableOp?
lstm/lstm_1/lstm_cell/split_1Split0lstm/lstm_1/lstm_cell/split_1/split_dim:output:04lstm/lstm_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm/lstm_1/lstm_cell/split_1?
lstm/lstm_1/lstm_cell/BiasAddBiasAdd&lstm/lstm_1/lstm_cell/MatMul:product:0&lstm/lstm_1/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/BiasAdd?
lstm/lstm_1/lstm_cell/BiasAdd_1BiasAdd(lstm/lstm_1/lstm_cell/MatMul_1:product:0&lstm/lstm_1/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2!
lstm/lstm_1/lstm_cell/BiasAdd_1?
lstm/lstm_1/lstm_cell/BiasAdd_2BiasAdd(lstm/lstm_1/lstm_cell/MatMul_2:product:0&lstm/lstm_1/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2!
lstm/lstm_1/lstm_cell/BiasAdd_2?
lstm/lstm_1/lstm_cell/BiasAdd_3BiasAdd(lstm/lstm_1/lstm_cell/MatMul_3:product:0&lstm/lstm_1/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2!
lstm/lstm_1/lstm_cell/BiasAdd_3?
lstm/lstm_1/lstm_cell/mul_4Mullstm/lstm_1/zeros:output:0*lstm/lstm_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/mul_4?
lstm/lstm_1/lstm_cell/mul_5Mullstm/lstm_1/zeros:output:0*lstm/lstm_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/mul_5?
lstm/lstm_1/lstm_cell/mul_6Mullstm/lstm_1/zeros:output:0*lstm/lstm_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/mul_6?
lstm/lstm_1/lstm_cell/mul_7Mullstm/lstm_1/zeros:output:0*lstm/lstm_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/mul_7?
$lstm/lstm_1/lstm_cell/ReadVariableOpReadVariableOp-lstm_lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm/lstm_1/lstm_cell/ReadVariableOp?
)lstm/lstm_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm/lstm_1/lstm_cell/strided_slice/stack?
+lstm/lstm_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm/lstm_1/lstm_cell/strided_slice/stack_1?
+lstm/lstm_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm/lstm_1/lstm_cell/strided_slice/stack_2?
#lstm/lstm_1/lstm_cell/strided_sliceStridedSlice,lstm/lstm_1/lstm_cell/ReadVariableOp:value:02lstm/lstm_1/lstm_cell/strided_slice/stack:output:04lstm/lstm_1/lstm_cell/strided_slice/stack_1:output:04lstm/lstm_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2%
#lstm/lstm_1/lstm_cell/strided_slice?
lstm/lstm_1/lstm_cell/MatMul_4MatMullstm/lstm_1/lstm_cell/mul_4:z:0,lstm/lstm_1/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2 
lstm/lstm_1/lstm_cell/MatMul_4?
lstm/lstm_1/lstm_cell/addAddV2&lstm/lstm_1/lstm_cell/BiasAdd:output:0(lstm/lstm_1/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/add?
lstm/lstm_1/lstm_cell/SigmoidSigmoidlstm/lstm_1/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/Sigmoid?
&lstm/lstm_1/lstm_cell/ReadVariableOp_1ReadVariableOp-lstm_lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&lstm/lstm_1/lstm_cell/ReadVariableOp_1?
+lstm/lstm_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm/lstm_1/lstm_cell/strided_slice_1/stack?
-lstm/lstm_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm/lstm_1/lstm_cell/strided_slice_1/stack_1?
-lstm/lstm_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-lstm/lstm_1/lstm_cell/strided_slice_1/stack_2?
%lstm/lstm_1/lstm_cell/strided_slice_1StridedSlice.lstm/lstm_1/lstm_cell/ReadVariableOp_1:value:04lstm/lstm_1/lstm_cell/strided_slice_1/stack:output:06lstm/lstm_1/lstm_cell/strided_slice_1/stack_1:output:06lstm/lstm_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2'
%lstm/lstm_1/lstm_cell/strided_slice_1?
lstm/lstm_1/lstm_cell/MatMul_5MatMullstm/lstm_1/lstm_cell/mul_5:z:0.lstm/lstm_1/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2 
lstm/lstm_1/lstm_cell/MatMul_5?
lstm/lstm_1/lstm_cell/add_1AddV2(lstm/lstm_1/lstm_cell/BiasAdd_1:output:0(lstm/lstm_1/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/add_1?
lstm/lstm_1/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_1/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2!
lstm/lstm_1/lstm_cell/Sigmoid_1?
lstm/lstm_1/lstm_cell/mul_8Mul#lstm/lstm_1/lstm_cell/Sigmoid_1:y:0lstm/lstm_1/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/mul_8?
&lstm/lstm_1/lstm_cell/ReadVariableOp_2ReadVariableOp-lstm_lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&lstm/lstm_1/lstm_cell/ReadVariableOp_2?
+lstm/lstm_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm/lstm_1/lstm_cell/strided_slice_2/stack?
-lstm/lstm_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-lstm/lstm_1/lstm_cell/strided_slice_2/stack_1?
-lstm/lstm_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-lstm/lstm_1/lstm_cell/strided_slice_2/stack_2?
%lstm/lstm_1/lstm_cell/strided_slice_2StridedSlice.lstm/lstm_1/lstm_cell/ReadVariableOp_2:value:04lstm/lstm_1/lstm_cell/strided_slice_2/stack:output:06lstm/lstm_1/lstm_cell/strided_slice_2/stack_1:output:06lstm/lstm_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2'
%lstm/lstm_1/lstm_cell/strided_slice_2?
lstm/lstm_1/lstm_cell/MatMul_6MatMullstm/lstm_1/lstm_cell/mul_6:z:0.lstm/lstm_1/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2 
lstm/lstm_1/lstm_cell/MatMul_6?
lstm/lstm_1/lstm_cell/add_2AddV2(lstm/lstm_1/lstm_cell/BiasAdd_2:output:0(lstm/lstm_1/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/add_2?
lstm/lstm_1/lstm_cell/TanhTanhlstm/lstm_1/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/Tanh?
lstm/lstm_1/lstm_cell/mul_9Mul!lstm/lstm_1/lstm_cell/Sigmoid:y:0lstm/lstm_1/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/mul_9?
lstm/lstm_1/lstm_cell/add_3AddV2lstm/lstm_1/lstm_cell/mul_8:z:0lstm/lstm_1/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/add_3?
&lstm/lstm_1/lstm_cell/ReadVariableOp_3ReadVariableOp-lstm_lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&lstm/lstm_1/lstm_cell/ReadVariableOp_3?
+lstm/lstm_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+lstm/lstm_1/lstm_cell/strided_slice_3/stack?
-lstm/lstm_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2/
-lstm/lstm_1/lstm_cell/strided_slice_3/stack_1?
-lstm/lstm_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-lstm/lstm_1/lstm_cell/strided_slice_3/stack_2?
%lstm/lstm_1/lstm_cell/strided_slice_3StridedSlice.lstm/lstm_1/lstm_cell/ReadVariableOp_3:value:04lstm/lstm_1/lstm_cell/strided_slice_3/stack:output:06lstm/lstm_1/lstm_cell/strided_slice_3/stack_1:output:06lstm/lstm_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2'
%lstm/lstm_1/lstm_cell/strided_slice_3?
lstm/lstm_1/lstm_cell/MatMul_7MatMullstm/lstm_1/lstm_cell/mul_7:z:0.lstm/lstm_1/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2 
lstm/lstm_1/lstm_cell/MatMul_7?
lstm/lstm_1/lstm_cell/add_4AddV2(lstm/lstm_1/lstm_cell/BiasAdd_3:output:0(lstm/lstm_1/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/add_4?
lstm/lstm_1/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_1/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2!
lstm/lstm_1/lstm_cell/Sigmoid_2?
lstm/lstm_1/lstm_cell/Tanh_1Tanhlstm/lstm_1/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/Tanh_1?
lstm/lstm_1/lstm_cell/mul_10Mul#lstm/lstm_1/lstm_cell/Sigmoid_2:y:0 lstm/lstm_1/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/lstm_cell/mul_10?
)lstm/lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2+
)lstm/lstm_1/TensorArrayV2_1/element_shape?
lstm/lstm_1/TensorArrayV2_1TensorListReserve2lstm/lstm_1/TensorArrayV2_1/element_shape:output:0$lstm/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/lstm_1/TensorArrayV2_1f
lstm/lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/lstm_1/time?
)lstm/lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)lstm/lstm_1/TensorArrayV2_2/element_shape?
lstm/lstm_1/TensorArrayV2_2TensorListReserve2lstm/lstm_1/TensorArrayV2_2/element_shape:output:0$lstm/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
lstm/lstm_1/TensorArrayV2_2?
Clstm/lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2E
Clstm/lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
5lstm/lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm/lstm_1/transpose_1:y:0Llstm/lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type027
5lstm/lstm_1/TensorArrayUnstack_1/TensorListFromTensor?
lstm/lstm_1/zeros_like	ZerosLike lstm/lstm_1/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2
lstm/lstm_1/zeros_like?
$lstm/lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$lstm/lstm_1/while/maximum_iterations?
lstm/lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm/lstm_1/while/loop_counter?
lstm/lstm_1/whileWhile'lstm/lstm_1/while/loop_counter:output:0-lstm/lstm_1/while/maximum_iterations:output:0lstm/lstm_1/time:output:0$lstm/lstm_1/TensorArrayV2_1:handle:0lstm/lstm_1/zeros_like:y:0lstm/lstm_1/zeros:output:0lstm/lstm_1/zeros_1:output:0$lstm/lstm_1/strided_slice_1:output:0Clstm/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Elstm/lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:03lstm_lstm_1_lstm_cell_split_readvariableop_resource5lstm_lstm_1_lstm_cell_split_1_readvariableop_resource-lstm_lstm_1_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *(
body R
lstm_lstm_1_while_body_17305*(
cond R
lstm_lstm_1_while_cond_17304*c
output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *
parallel_iterations 2
lstm/lstm_1/while?
<lstm/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<lstm/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape?
.lstm/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm/lstm_1/while:output:3Elstm/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype020
.lstm/lstm_1/TensorArrayV2Stack/TensorListStack?
!lstm/lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2#
!lstm/lstm_1/strided_slice_3/stack?
#lstm/lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#lstm/lstm_1/strided_slice_3/stack_1?
#lstm/lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#lstm/lstm_1/strided_slice_3/stack_2?
lstm/lstm_1/strided_slice_3StridedSlice7lstm/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0*lstm/lstm_1/strided_slice_3/stack:output:0,lstm/lstm_1/strided_slice_3/stack_1:output:0,lstm/lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm/lstm_1/strided_slice_3?
lstm/lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/lstm_1/transpose_2/perm?
lstm/lstm_1/transpose_2	Transpose7lstm/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm/lstm_1/transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm/lstm_1/transpose_2~
lstm/lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/lstm_1/runtimef
lstm/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/concat/axis?
lstm/concatConcatV2$lstm/lstm_1/strided_slice_3:output:0lstm/lstm_1/while:output:5lstm/lstm_1/while:output:6lstm/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
lstm/concat?
 lstm/dense/MatMul/ReadVariableOpReadVariableOp)lstm_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 lstm/dense/MatMul/ReadVariableOp?
lstm/dense/MatMulMatMullstm/concat:output:0(lstm/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm/dense/MatMul?
!lstm/dense/BiasAdd/ReadVariableOpReadVariableOp*lstm_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!lstm/dense/BiasAdd/ReadVariableOp?
lstm/dense/BiasAddBiasAddlstm/dense/MatMul:product:0)lstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm/dense/BiasAdd?
lstm/dense/LeakyRelu	LeakyRelulstm/dense/BiasAdd:output:0*(
_output_shapes
:??????????2
lstm/dense/LeakyRelu?
lstm/dropout/IdentityIdentity"lstm/dense/LeakyRelu:activations:0*
T0*(
_output_shapes
:??????????2
lstm/dropout/Identity?
"lstm/dense_1/MatMul/ReadVariableOpReadVariableOp+lstm_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm/dense_1/MatMul/ReadVariableOp?
lstm/dense_1/MatMulMatMullstm/dropout/Identity:output:0*lstm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
lstm/dense_1/MatMul?
#lstm/dense_1/BiasAdd/ReadVariableOpReadVariableOp,lstm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#lstm/dense_1/BiasAdd/ReadVariableOp?
lstm/dense_1/BiasAddBiasAddlstm/dense_1/MatMul:product:0+lstm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
lstm/dense_1/BiasAddx
IdentityIdentitylstm/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^lstm/dense/BiasAdd/ReadVariableOp!^lstm/dense/MatMul/ReadVariableOp$^lstm/dense_1/BiasAdd/ReadVariableOp#^lstm/dense_1/MatMul/ReadVariableOp%^lstm/lstm_1/lstm_cell/ReadVariableOp'^lstm/lstm_1/lstm_cell/ReadVariableOp_1'^lstm/lstm_1/lstm_cell/ReadVariableOp_2'^lstm/lstm_1/lstm_cell/ReadVariableOp_3+^lstm/lstm_1/lstm_cell/split/ReadVariableOp-^lstm/lstm_1/lstm_cell/split_1/ReadVariableOp^lstm/lstm_1/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 2F
!lstm/dense/BiasAdd/ReadVariableOp!lstm/dense/BiasAdd/ReadVariableOp2D
 lstm/dense/MatMul/ReadVariableOp lstm/dense/MatMul/ReadVariableOp2J
#lstm/dense_1/BiasAdd/ReadVariableOp#lstm/dense_1/BiasAdd/ReadVariableOp2H
"lstm/dense_1/MatMul/ReadVariableOp"lstm/dense_1/MatMul/ReadVariableOp2L
$lstm/lstm_1/lstm_cell/ReadVariableOp$lstm/lstm_1/lstm_cell/ReadVariableOp2P
&lstm/lstm_1/lstm_cell/ReadVariableOp_1&lstm/lstm_1/lstm_cell/ReadVariableOp_12P
&lstm/lstm_1/lstm_cell/ReadVariableOp_2&lstm/lstm_1/lstm_cell/ReadVariableOp_22P
&lstm/lstm_1/lstm_cell/ReadVariableOp_3&lstm/lstm_1/lstm_cell/ReadVariableOp_32X
*lstm/lstm_1/lstm_cell/split/ReadVariableOp*lstm/lstm_1/lstm_cell/split/ReadVariableOp2\
,lstm/lstm_1/lstm_cell/split_1/ReadVariableOp,lstm/lstm_1/lstm_cell/split_1/ReadVariableOp2&
lstm/lstm_1/whilelstm/lstm_1/while:T P
+
_output_shapes
:?????????E
!
_user_specified_name	input_1
??
?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_17864

inputs

states
states_10
split_readvariableop_resource:	E?.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????E2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2Ӎ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????E2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????E2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????E2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????E2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2??2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????E2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????E2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????E2
dropout_3/Mul_1\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/Const?
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ޜ2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_5/Const?
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_6/Const?
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shape?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2٪?2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_6/GreaterEqual/y?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/GreaterEqual?
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_6/Cast?
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_7/Const?
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shape?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_7/GreaterEqual/y?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/GreaterEqual?
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_7/Cast?
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
mul_3d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	E?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3e
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_4e
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_5e
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_6e
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10f
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????E:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????E
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
lstm_1_while_cond_19439*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4,
(lstm_1_while_less_lstm_1_strided_slice_1A
=lstm_1_while_lstm_1_while_cond_19439___redundant_placeholder0A
=lstm_1_while_lstm_1_while_cond_19439___redundant_placeholder1A
=lstm_1_while_lstm_1_while_cond_19439___redundant_placeholder2A
=lstm_1_while_lstm_1_while_cond_19439___redundant_placeholder3A
=lstm_1_while_lstm_1_while_cond_19439___redundant_placeholder4
lstm_1_while_identity
?
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: 2
lstm_1/while/Lessr
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_1/while/Identity"7
lstm_1_while_identitylstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :??????????:??????????:??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?
?
?__inference_lstm_layer_call_and_return_conditional_losses_19283
input_1
lstm_1_19260:	E?
lstm_1_19262:	? 
lstm_1_19264:
??
dense_19271:
??
dense_19273:	? 
dense_1_19277:	?
dense_1_19279:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?lstm_1/StatefulPartitionedCallK
xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
x?
NotEqualNotEqualinput_1
x:output:0*
T0*+
_output_shapes
:?????????E*
incompatible_shape_error( 2

NotEqualp
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Any/reduction_indicesh
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*'
_output_shapes
:?????????2
Any?
lstm_1/StatefulPartitionedCallStatefulPartitionedCallinput_1Any:output:0lstm_1_19260lstm_1_19262lstm_1_19264*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_191232 
lstm_1/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2'lstm_1/StatefulPartitionedCall:output:0'lstm_1/StatefulPartitionedCall:output:1'lstm_1/StatefulPartitionedCall:output:2concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_19271dense_19273*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_186052
dense/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_186822!
dropout/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_19277dense_1_19279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_186282!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:T P
+
_output_shapes
:?????????E
!
_user_specified_name	input_1
?
C
'__inference_dropout_layer_call_fn_21526

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_186162
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_20195
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_20195___redundant_placeholder03
/while_while_cond_20195___redundant_placeholder13
/while_while_cond_20195___redundant_placeholder23
/while_while_cond_20195___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
?__inference_lstm_layer_call_and_return_conditional_losses_19253
input_1
lstm_1_19230:	E?
lstm_1_19232:	? 
lstm_1_19234:
??
dense_19241:
??
dense_19243:	? 
dense_1_19247:	?
dense_1_19249:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?lstm_1/StatefulPartitionedCallK
xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
x?
NotEqualNotEqualinput_1
x:output:0*
T0*+
_output_shapes
:?????????E*
incompatible_shape_error( 2

NotEqualp
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Any/reduction_indicesh
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*'
_output_shapes
:?????????2
Any?
lstm_1/StatefulPartitionedCallStatefulPartitionedCallinput_1Any:output:0lstm_1_19230lstm_1_19232lstm_1_19234*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_185822 
lstm_1/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2'lstm_1/StatefulPartitionedCall:output:0'lstm_1/StatefulPartitionedCall:output:1'lstm_1/StatefulPartitionedCall:output:2concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_19241dense_19243*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_186052
dense/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_186162
dropout/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_19247dense_1_19249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_186282!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:T P
+
_output_shapes
:?????????E
!
_user_specified_name	input_1
?

?
while_cond_20839
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_20839___redundant_placeholder03
/while_while_cond_20839___redundant_placeholder13
/while_while_cond_20839___redundant_placeholder23
/while_while_cond_20839___redundant_placeholder33
/while_while_cond_20839___redundant_placeholder4
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :??????????:??????????:??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?
?
lstm_lstm_1_while_cond_173044
0lstm_lstm_1_while_lstm_lstm_1_while_loop_counter:
6lstm_lstm_1_while_lstm_lstm_1_while_maximum_iterations!
lstm_lstm_1_while_placeholder#
lstm_lstm_1_while_placeholder_1#
lstm_lstm_1_while_placeholder_2#
lstm_lstm_1_while_placeholder_3#
lstm_lstm_1_while_placeholder_46
2lstm_lstm_1_while_less_lstm_lstm_1_strided_slice_1K
Glstm_lstm_1_while_lstm_lstm_1_while_cond_17304___redundant_placeholder0K
Glstm_lstm_1_while_lstm_lstm_1_while_cond_17304___redundant_placeholder1K
Glstm_lstm_1_while_lstm_lstm_1_while_cond_17304___redundant_placeholder2K
Glstm_lstm_1_while_lstm_lstm_1_while_cond_17304___redundant_placeholder3K
Glstm_lstm_1_while_lstm_lstm_1_while_cond_17304___redundant_placeholder4
lstm_lstm_1_while_identity
?
lstm/lstm_1/while/LessLesslstm_lstm_1_while_placeholder2lstm_lstm_1_while_less_lstm_lstm_1_strided_slice_1*
T0*
_output_shapes
: 2
lstm/lstm_1/while/Less?
lstm/lstm_1/while/IdentityIdentitylstm/lstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/lstm_1/while/Identity"A
lstm_lstm_1_while_identity#lstm/lstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :??????????:??????????:??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?
?
@__inference_dense_layer_call_and_return_conditional_losses_18605

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:??????????2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_19123

inputs
mask
:
'lstm_cell_split_readvariableop_resource:	E?8
)lstm_cell_split_1_readvariableop_resource:	?5
!lstm_cell_readvariableop_resource:
??
identity

identity_1

identity_2??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????E2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim{

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:?????????2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:?????????2
transpose_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2??/20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2?ұ22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/dropout_3/Mul_1x
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??j22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??822
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_3x
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	E?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time?
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2_2/element_shape?
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2?
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensorn

zeros_like	ZerosLikelstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *
bodyR
while_body_18905*
condR
while_cond_18904*c
output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityn

Identity_1Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1n

Identity_2Identitywhile:output:6^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????E:?????????: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????E
 
_user_specified_nameinputs:MI
'
_output_shapes
:?????????

_user_specified_namemask
?	
?
#__inference_signature_wrapper_19310
input_1
unknown:	E?
	unknown_0:	?
	unknown_1:
??
	unknown_2:
??
	unknown_3:	?
	unknown_4:	?
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_174732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????E
!
_user_specified_name	input_1
?%
?
while_body_17946
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_17970_0:	E?&
while_lstm_cell_17972_0:	?+
while_lstm_cell_17974_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_17970:	E?$
while_lstm_cell_17972:	?)
while_lstm_cell_17974:
????'while/lstm_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_17970_0while_lstm_cell_17972_0while_lstm_cell_17974_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_178642)
'while/lstm_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_17970while_lstm_cell_17970_0"0
while_lstm_cell_17972while_lstm_cell_17972_0"0
while_lstm_cell_17974while_lstm_cell_17974_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?

while_body_18905
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	E?@
1while_lstm_cell_split_1_readvariableop_resource_0:	?=
)while_lstm_cell_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	E?>
/while_lstm_cell_split_1_readvariableop_resource:	?;
'while_lstm_cell_readvariableop_resource:
????while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem?
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape?
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/ones_like/Const?
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/ones_like?
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/lstm_cell/dropout/Const?
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout/Mul?
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/Shape?
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2إ26
4while/lstm_cell/dropout/random_uniform/RandomUniform?
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2(
&while/lstm_cell/dropout/GreaterEqual/y?
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2&
$while/lstm_cell/dropout/GreaterEqual?
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2
while/lstm_cell/dropout/Cast?
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout/Mul_1?
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_1/Const?
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout_1/Mul?
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/Shape?
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_1/random_uniform/RandomUniform?
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_1/GreaterEqual/y?
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2(
&while/lstm_cell/dropout_1/GreaterEqual?
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2 
while/lstm_cell/dropout_1/Cast?
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????E2!
while/lstm_cell/dropout_1/Mul_1?
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_2/Const?
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout_2/Mul?
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/Shape?
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_2/random_uniform/RandomUniform?
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_2/GreaterEqual/y?
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2(
&while/lstm_cell/dropout_2/GreaterEqual?
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2 
while/lstm_cell/dropout_2/Cast?
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????E2!
while/lstm_cell/dropout_2/Mul_1?
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_3/Const?
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/dropout_3/Mul?
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/Shape?
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2??u28
6while/lstm_cell/dropout_3/random_uniform/RandomUniform?
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_3/GreaterEqual/y?
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2(
&while/lstm_cell/dropout_3/GreaterEqual?
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2 
while/lstm_cell/dropout_3/Cast?
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????E2!
while/lstm_cell/dropout_3/Mul_1?
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape?
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell/ones_like_1/Const?
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/ones_like_1?
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_4/Const?
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_4/Mul?
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_4/Shape?
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_4/random_uniform/RandomUniform?
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_4/GreaterEqual/y?
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_4/GreaterEqual?
while/lstm_cell/dropout_4/CastCast*while/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_4/Cast?
while/lstm_cell/dropout_4/Mul_1Mul!while/lstm_cell/dropout_4/Mul:z:0"while/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_4/Mul_1?
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_5/Const?
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_5/Mul?
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_5/Shape?
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_5/random_uniform/RandomUniform?
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_5/GreaterEqual/y?
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_5/GreaterEqual?
while/lstm_cell/dropout_5/CastCast*while/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_5/Cast?
while/lstm_cell/dropout_5/Mul_1Mul!while/lstm_cell/dropout_5/Mul:z:0"while/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_5/Mul_1?
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_6/Const?
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_6/Mul?
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_6/Shape?
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_6/random_uniform/RandomUniform?
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_6/GreaterEqual/y?
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_6/GreaterEqual?
while/lstm_cell/dropout_6/CastCast*while/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_6/Cast?
while/lstm_cell/dropout_6/Mul_1Mul!while/lstm_cell/dropout_6/Mul:z:0"while/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_6/Mul_1?
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/lstm_cell/dropout_7/Const?
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/dropout_7/Mul?
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_7/Shape?
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell/dropout_7/random_uniform/RandomUniform?
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(while/lstm_cell/dropout_7/GreaterEqual/y?
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&while/lstm_cell/dropout_7/GreaterEqual?
while/lstm_cell/dropout_7/CastCast*while/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
while/lstm_cell/dropout_7/Cast?
while/lstm_cell/dropout_7/Mul_1Mul!while/lstm_cell/dropout_7/Mul:z:0"while/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/dropout_7/Mul_1?
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul?
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_1?
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_2?
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
while/lstm_cell/mul_3?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	E?*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/mul_4Mulwhile_placeholder_3#while/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_4?
while/lstm_cell/mul_5Mulwhile_placeholder_3#while/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
while/lstm_cell/mul_6Mulwhile_placeholder_3#while/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_6?
while/lstm_cell/mul_7Mulwhile_placeholder_3#while/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_7?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_4*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_8?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2?
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh?
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_9?
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_3?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Tanh_1?
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_10}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2

while/Tile?
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell/mul_10:z:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/SelectV2?
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_1?
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_2?
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell/mul_10:z:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/SelectV2_1?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell/add_3:z:0while_placeholder_4*
T0*(
_output_shapes
:??????????2
while/SelectV2_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?
while/Identity_6Identitywhile/SelectV2_2:output:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_6?

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :??????????:??????????:??????????: : : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_18682

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
?__inference_lstm_layer_call_and_return_conditional_losses_19608
xA
.lstm_1_lstm_cell_split_readvariableop_resource:	E??
0lstm_1_lstm_cell_split_1_readvariableop_resource:	?<
(lstm_1_lstm_cell_readvariableop_resource:
??8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?lstm_1/lstm_cell/ReadVariableOp?!lstm_1/lstm_cell/ReadVariableOp_1?!lstm_1/lstm_cell/ReadVariableOp_2?!lstm_1/lstm_cell/ReadVariableOp_3?%lstm_1/lstm_cell/split/ReadVariableOp?'lstm_1/lstm_cell/split_1/ReadVariableOp?lstm_1/whileO
x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
x_1?
NotEqualNotEqualxx_1:output:0*
T0*+
_output_shapes
:?????????E*
incompatible_shape_error( 2

NotEqualp
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Any/reduction_indicesh
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*'
_output_shapes
:?????????2
AnyM
lstm_1/ShapeShapex*
T0*
_output_shapes
:2
lstm_1/Shape?
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice/stack?
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_1?
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_2?
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slicek
lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros/mul/y?
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/mulm
lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros/Less/y?
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/Lessq
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros/packed/1?
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros/packedm
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros/Const?
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/zeroso
lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros_1/mul/y?
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/mulq
lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros_1/Less/y?
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/Lessu
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros_1/packed/1?
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros_1/packedq
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros_1/Const?
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/zeros_1?
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose/perm?
lstm_1/transpose	Transposexlstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????E2
lstm_1/transposed
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:2
lstm_1/Shape_1?
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_1/stack?
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_1?
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_2?
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slice_1y
lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_1/ExpandDims/dim?
lstm_1/ExpandDims
ExpandDimsAny:output:0lstm_1/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:?????????2
lstm_1/ExpandDims?
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_1/perm?
lstm_1/transpose_1	Transposelstm_1/ExpandDims:output:0 lstm_1/transpose_1/perm:output:0*
T0
*+
_output_shapes
:?????????2
lstm_1/transpose_1?
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_1/TensorArrayV2/element_shape?
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2?
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   2>
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_1/TensorArrayUnstack/TensorListFromTensor?
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_2/stack?
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_1?
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_2?
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
lstm_1/strided_slice_2?
 lstm_1/lstm_cell/ones_like/ShapeShapelstm_1/strided_slice_2:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell/ones_like/Shape?
 lstm_1/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 lstm_1/lstm_cell/ones_like/Const?
lstm_1/lstm_cell/ones_likeFill)lstm_1/lstm_cell/ones_like/Shape:output:0)lstm_1/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/ones_like?
"lstm_1/lstm_cell/ones_like_1/ShapeShapelstm_1/zeros:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell/ones_like_1/Shape?
"lstm_1/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"lstm_1/lstm_cell/ones_like_1/Const?
lstm_1/lstm_cell/ones_like_1Fill+lstm_1/lstm_cell/ones_like_1/Shape:output:0+lstm_1/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/ones_like_1?
lstm_1/lstm_cell/mulMullstm_1/strided_slice_2:output:0#lstm_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/mul?
lstm_1/lstm_cell/mul_1Mullstm_1/strided_slice_2:output:0#lstm_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/mul_1?
lstm_1/lstm_cell/mul_2Mullstm_1/strided_slice_2:output:0#lstm_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/mul_2?
lstm_1/lstm_cell/mul_3Mullstm_1/strided_slice_2:output:0#lstm_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/lstm_cell/mul_3?
 lstm_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_1/lstm_cell/split/split_dim?
%lstm_1/lstm_cell/split/ReadVariableOpReadVariableOp.lstm_1_lstm_cell_split_readvariableop_resource*
_output_shapes
:	E?*
dtype02'
%lstm_1/lstm_cell/split/ReadVariableOp?
lstm_1/lstm_cell/splitSplit)lstm_1/lstm_cell/split/split_dim:output:0-lstm_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm_1/lstm_cell/split?
lstm_1/lstm_cell/MatMulMatMullstm_1/lstm_cell/mul:z:0lstm_1/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul?
lstm_1/lstm_cell/MatMul_1MatMullstm_1/lstm_cell/mul_1:z:0lstm_1/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_1?
lstm_1/lstm_cell/MatMul_2MatMullstm_1/lstm_cell/mul_2:z:0lstm_1/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_2?
lstm_1/lstm_cell/MatMul_3MatMullstm_1/lstm_cell/mul_3:z:0lstm_1/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_3?
"lstm_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lstm_1/lstm_cell/split_1/split_dim?
'lstm_1/lstm_cell/split_1/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'lstm_1/lstm_cell/split_1/ReadVariableOp?
lstm_1/lstm_cell/split_1Split+lstm_1/lstm_cell/split_1/split_dim:output:0/lstm_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_1/lstm_cell/split_1?
lstm_1/lstm_cell/BiasAddBiasAdd!lstm_1/lstm_cell/MatMul:product:0!lstm_1/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/BiasAdd?
lstm_1/lstm_cell/BiasAdd_1BiasAdd#lstm_1/lstm_cell/MatMul_1:product:0!lstm_1/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/BiasAdd_1?
lstm_1/lstm_cell/BiasAdd_2BiasAdd#lstm_1/lstm_cell/MatMul_2:product:0!lstm_1/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/BiasAdd_2?
lstm_1/lstm_cell/BiasAdd_3BiasAdd#lstm_1/lstm_cell/MatMul_3:product:0!lstm_1/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/BiasAdd_3?
lstm_1/lstm_cell/mul_4Mullstm_1/zeros:output:0%lstm_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_4?
lstm_1/lstm_cell/mul_5Mullstm_1/zeros:output:0%lstm_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_5?
lstm_1/lstm_cell/mul_6Mullstm_1/zeros:output:0%lstm_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_6?
lstm_1/lstm_cell/mul_7Mullstm_1/zeros:output:0%lstm_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_7?
lstm_1/lstm_cell/ReadVariableOpReadVariableOp(lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
lstm_1/lstm_cell/ReadVariableOp?
$lstm_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_1/lstm_cell/strided_slice/stack?
&lstm_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&lstm_1/lstm_cell/strided_slice/stack_1?
&lstm_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm_1/lstm_cell/strided_slice/stack_2?
lstm_1/lstm_cell/strided_sliceStridedSlice'lstm_1/lstm_cell/ReadVariableOp:value:0-lstm_1/lstm_cell/strided_slice/stack:output:0/lstm_1/lstm_cell/strided_slice/stack_1:output:0/lstm_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
lstm_1/lstm_cell/strided_slice?
lstm_1/lstm_cell/MatMul_4MatMullstm_1/lstm_cell/mul_4:z:0'lstm_1/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_4?
lstm_1/lstm_cell/addAddV2!lstm_1/lstm_cell/BiasAdd:output:0#lstm_1/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/add?
lstm_1/lstm_cell/SigmoidSigmoidlstm_1/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/Sigmoid?
!lstm_1/lstm_cell/ReadVariableOp_1ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_1/lstm_cell/ReadVariableOp_1?
&lstm_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2(
&lstm_1/lstm_cell/strided_slice_1/stack?
(lstm_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell/strided_slice_1/stack_1?
(lstm_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell/strided_slice_1/stack_2?
 lstm_1/lstm_cell/strided_slice_1StridedSlice)lstm_1/lstm_cell/ReadVariableOp_1:value:0/lstm_1/lstm_cell/strided_slice_1/stack:output:01lstm_1/lstm_cell/strided_slice_1/stack_1:output:01lstm_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 lstm_1/lstm_cell/strided_slice_1?
lstm_1/lstm_cell/MatMul_5MatMullstm_1/lstm_cell/mul_5:z:0)lstm_1/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_5?
lstm_1/lstm_cell/add_1AddV2#lstm_1/lstm_cell/BiasAdd_1:output:0#lstm_1/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/add_1?
lstm_1/lstm_cell/Sigmoid_1Sigmoidlstm_1/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/Sigmoid_1?
lstm_1/lstm_cell/mul_8Mullstm_1/lstm_cell/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_8?
!lstm_1/lstm_cell/ReadVariableOp_2ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_1/lstm_cell/ReadVariableOp_2?
&lstm_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2(
&lstm_1/lstm_cell/strided_slice_2/stack?
(lstm_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell/strided_slice_2/stack_1?
(lstm_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell/strided_slice_2/stack_2?
 lstm_1/lstm_cell/strided_slice_2StridedSlice)lstm_1/lstm_cell/ReadVariableOp_2:value:0/lstm_1/lstm_cell/strided_slice_2/stack:output:01lstm_1/lstm_cell/strided_slice_2/stack_1:output:01lstm_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 lstm_1/lstm_cell/strided_slice_2?
lstm_1/lstm_cell/MatMul_6MatMullstm_1/lstm_cell/mul_6:z:0)lstm_1/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_6?
lstm_1/lstm_cell/add_2AddV2#lstm_1/lstm_cell/BiasAdd_2:output:0#lstm_1/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/add_2?
lstm_1/lstm_cell/TanhTanhlstm_1/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/Tanh?
lstm_1/lstm_cell/mul_9Mullstm_1/lstm_cell/Sigmoid:y:0lstm_1/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_9?
lstm_1/lstm_cell/add_3AddV2lstm_1/lstm_cell/mul_8:z:0lstm_1/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/add_3?
!lstm_1/lstm_cell/ReadVariableOp_3ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_1/lstm_cell/ReadVariableOp_3?
&lstm_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2(
&lstm_1/lstm_cell/strided_slice_3/stack?
(lstm_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm_1/lstm_cell/strided_slice_3/stack_1?
(lstm_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell/strided_slice_3/stack_2?
 lstm_1/lstm_cell/strided_slice_3StridedSlice)lstm_1/lstm_cell/ReadVariableOp_3:value:0/lstm_1/lstm_cell/strided_slice_3/stack:output:01lstm_1/lstm_cell/strided_slice_3/stack_1:output:01lstm_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 lstm_1/lstm_cell/strided_slice_3?
lstm_1/lstm_cell/MatMul_7MatMullstm_1/lstm_cell/mul_7:z:0)lstm_1/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/MatMul_7?
lstm_1/lstm_cell/add_4AddV2#lstm_1/lstm_cell/BiasAdd_3:output:0#lstm_1/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/add_4?
lstm_1/lstm_cell/Sigmoid_2Sigmoidlstm_1/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/Sigmoid_2?
lstm_1/lstm_cell/Tanh_1Tanhlstm_1/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/Tanh_1?
lstm_1/lstm_cell/mul_10Mullstm_1/lstm_cell/Sigmoid_2:y:0lstm_1/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_1/lstm_cell/mul_10?
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$lstm_1/TensorArrayV2_1/element_shape?
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2_1\
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/time?
$lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$lstm_1/TensorArrayV2_2/element_shape?
lstm_1/TensorArrayV2_2TensorListReserve-lstm_1/TensorArrayV2_2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
lstm_1/TensorArrayV2_2?
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
0lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm_1/transpose_1:y:0Glstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type022
0lstm_1/TensorArrayUnstack_1/TensorListFromTensor?
lstm_1/zeros_like	ZerosLikelstm_1/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/zeros_like?
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_1/while/maximum_iterationsx
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/while/loop_counter?
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros_like:y:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0.lstm_1_lstm_cell_split_readvariableop_resource0lstm_1_lstm_cell_split_1_readvariableop_resource(lstm_1_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *#
bodyR
lstm_1_while_body_19440*#
condR
lstm_1_while_cond_19439*c
output_shapesR
P: : : : :??????????:??????????:??????????: : : : : : *
parallel_iterations 2
lstm_1/while?
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)lstm_1/TensorArrayV2Stack/TensorListStack?
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_1/strided_slice_3/stack?
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_1/strided_slice_3/stack_1?
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_3/stack_2?
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_1/strided_slice_3?
lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_2/perm?
lstm_1/transpose_2	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_1/transpose_2t
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2lstm_1/strided_slice_3:output:0lstm_1/while:output:5lstm_1/while:output:6concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddq
dense/LeakyRelu	LeakyReludense/BiasAdd:output:0*(
_output_shapes
:??????????2
dense/LeakyRelu?
dropout/IdentityIdentitydense/LeakyRelu:activations:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdds
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^lstm_1/lstm_cell/ReadVariableOp"^lstm_1/lstm_cell/ReadVariableOp_1"^lstm_1/lstm_cell/ReadVariableOp_2"^lstm_1/lstm_cell/ReadVariableOp_3&^lstm_1/lstm_cell/split/ReadVariableOp(^lstm_1/lstm_cell/split_1/ReadVariableOp^lstm_1/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
lstm_1/lstm_cell/ReadVariableOplstm_1/lstm_cell/ReadVariableOp2F
!lstm_1/lstm_cell/ReadVariableOp_1!lstm_1/lstm_cell/ReadVariableOp_12F
!lstm_1/lstm_cell/ReadVariableOp_2!lstm_1/lstm_cell/ReadVariableOp_22F
!lstm_1/lstm_cell/ReadVariableOp_3!lstm_1/lstm_cell/ReadVariableOp_32N
%lstm_1/lstm_cell/split/ReadVariableOp%lstm_1/lstm_cell/split/ReadVariableOp2R
'lstm_1/lstm_cell/split_1/ReadVariableOp'lstm_1/lstm_cell/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:N J
+
_output_shapes
:?????????E

_user_specified_namex
?
?
&__inference_lstm_1_layer_call_fn_21433
inputs_0
unknown:	E?
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_180172
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????E: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????E
"
_user_specified_name
inputs/0
??
?
lstm_1_while_body_19802*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0i
elstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0I
6lstm_1_while_lstm_cell_split_readvariableop_resource_0:	E?G
8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0:	?D
0lstm_1_while_lstm_cell_readvariableop_resource_0:
??
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5
lstm_1_while_identity_6'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorg
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensorG
4lstm_1_while_lstm_cell_split_readvariableop_resource:	E?E
6lstm_1_while_lstm_cell_split_1_readvariableop_resource:	?B
.lstm_1_while_lstm_cell_readvariableop_resource:
????%lstm_1/while/lstm_cell/ReadVariableOp?'lstm_1/while/lstm_cell/ReadVariableOp_1?'lstm_1/while/lstm_cell/ReadVariableOp_2?'lstm_1/while/lstm_cell/ReadVariableOp_3?+lstm_1/while/lstm_cell/split/ReadVariableOp?-lstm_1/while/lstm_cell/split_1/ReadVariableOp?
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   2@
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype022
0lstm_1/while/TensorArrayV2Read/TensorListGetItem?
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0lstm_1_while_placeholderIlstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
24
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItem?
&lstm_1/while/lstm_cell/ones_like/ShapeShape7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell/ones_like/Shape?
&lstm_1/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&lstm_1/while/lstm_cell/ones_like/Const?
 lstm_1/while/lstm_cell/ones_likeFill/lstm_1/while/lstm_cell/ones_like/Shape:output:0/lstm_1/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_1/while/lstm_cell/ones_like?
$lstm_1/while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$lstm_1/while/lstm_cell/dropout/Const?
"lstm_1/while/lstm_cell/dropout/MulMul)lstm_1/while/lstm_cell/ones_like:output:0-lstm_1/while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????E2$
"lstm_1/while/lstm_cell/dropout/Mul?
$lstm_1/while/lstm_cell/dropout/ShapeShape)lstm_1/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm_1/while/lstm_cell/dropout/Shape?
;lstm_1/while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform-lstm_1/while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2?̲2=
;lstm_1/while/lstm_cell/dropout/random_uniform/RandomUniform?
-lstm_1/while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2/
-lstm_1/while/lstm_cell/dropout/GreaterEqual/y?
+lstm_1/while/lstm_cell/dropout/GreaterEqualGreaterEqualDlstm_1/while/lstm_cell/dropout/random_uniform/RandomUniform:output:06lstm_1/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2-
+lstm_1/while/lstm_cell/dropout/GreaterEqual?
#lstm_1/while/lstm_cell/dropout/CastCast/lstm_1/while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2%
#lstm_1/while/lstm_cell/dropout/Cast?
$lstm_1/while/lstm_cell/dropout/Mul_1Mul&lstm_1/while/lstm_cell/dropout/Mul:z:0'lstm_1/while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????E2&
$lstm_1/while/lstm_cell/dropout/Mul_1?
&lstm_1/while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&lstm_1/while/lstm_cell/dropout_1/Const?
$lstm_1/while/lstm_cell/dropout_1/MulMul)lstm_1/while/lstm_cell/ones_like:output:0/lstm_1/while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????E2&
$lstm_1/while/lstm_cell/dropout_1/Mul?
&lstm_1/while/lstm_cell/dropout_1/ShapeShape)lstm_1/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell/dropout_1/Shape?
=lstm_1/while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2???2?
=lstm_1/while/lstm_cell/dropout_1/random_uniform/RandomUniform?
/lstm_1/while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>21
/lstm_1/while/lstm_cell/dropout_1/GreaterEqual/y?
-lstm_1/while/lstm_cell/dropout_1/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2/
-lstm_1/while/lstm_cell/dropout_1/GreaterEqual?
%lstm_1/while/lstm_cell/dropout_1/CastCast1lstm_1/while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2'
%lstm_1/while/lstm_cell/dropout_1/Cast?
&lstm_1/while/lstm_cell/dropout_1/Mul_1Mul(lstm_1/while/lstm_cell/dropout_1/Mul:z:0)lstm_1/while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????E2(
&lstm_1/while/lstm_cell/dropout_1/Mul_1?
&lstm_1/while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&lstm_1/while/lstm_cell/dropout_2/Const?
$lstm_1/while/lstm_cell/dropout_2/MulMul)lstm_1/while/lstm_cell/ones_like:output:0/lstm_1/while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????E2&
$lstm_1/while/lstm_cell/dropout_2/Mul?
&lstm_1/while/lstm_cell/dropout_2/ShapeShape)lstm_1/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell/dropout_2/Shape?
=lstm_1/while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2?ҳ2?
=lstm_1/while/lstm_cell/dropout_2/random_uniform/RandomUniform?
/lstm_1/while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>21
/lstm_1/while/lstm_cell/dropout_2/GreaterEqual/y?
-lstm_1/while/lstm_cell/dropout_2/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2/
-lstm_1/while/lstm_cell/dropout_2/GreaterEqual?
%lstm_1/while/lstm_cell/dropout_2/CastCast1lstm_1/while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2'
%lstm_1/while/lstm_cell/dropout_2/Cast?
&lstm_1/while/lstm_cell/dropout_2/Mul_1Mul(lstm_1/while/lstm_cell/dropout_2/Mul:z:0)lstm_1/while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????E2(
&lstm_1/while/lstm_cell/dropout_2/Mul_1?
&lstm_1/while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&lstm_1/while/lstm_cell/dropout_3/Const?
$lstm_1/while/lstm_cell/dropout_3/MulMul)lstm_1/while/lstm_cell/ones_like:output:0/lstm_1/while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????E2&
$lstm_1/while/lstm_cell/dropout_3/Mul?
&lstm_1/while/lstm_cell/dropout_3/ShapeShape)lstm_1/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell/dropout_3/Shape?
=lstm_1/while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????E*
dtype0*
seed???)*
seed2?ǎ2?
=lstm_1/while/lstm_cell/dropout_3/random_uniform/RandomUniform?
/lstm_1/while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>21
/lstm_1/while/lstm_cell/dropout_3/GreaterEqual/y?
-lstm_1/while/lstm_cell/dropout_3/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????E2/
-lstm_1/while/lstm_cell/dropout_3/GreaterEqual?
%lstm_1/while/lstm_cell/dropout_3/CastCast1lstm_1/while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????E2'
%lstm_1/while/lstm_cell/dropout_3/Cast?
&lstm_1/while/lstm_cell/dropout_3/Mul_1Mul(lstm_1/while/lstm_cell/dropout_3/Mul:z:0)lstm_1/while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????E2(
&lstm_1/while/lstm_cell/dropout_3/Mul_1?
(lstm_1/while/lstm_cell/ones_like_1/ShapeShapelstm_1_while_placeholder_3*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell/ones_like_1/Shape?
(lstm_1/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(lstm_1/while/lstm_cell/ones_like_1/Const?
"lstm_1/while/lstm_cell/ones_like_1Fill1lstm_1/while/lstm_cell/ones_like_1/Shape:output:01lstm_1/while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_1/while/lstm_cell/ones_like_1?
&lstm_1/while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&lstm_1/while/lstm_cell/dropout_4/Const?
$lstm_1/while/lstm_cell/dropout_4/MulMul+lstm_1/while/lstm_cell/ones_like_1:output:0/lstm_1/while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2&
$lstm_1/while/lstm_cell/dropout_4/Mul?
&lstm_1/while/lstm_cell/dropout_4/ShapeShape+lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell/dropout_4/Shape?
=lstm_1/while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=lstm_1/while/lstm_cell/dropout_4/random_uniform/RandomUniform?
/lstm_1/while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>21
/lstm_1/while/lstm_cell/dropout_4/GreaterEqual/y?
-lstm_1/while/lstm_cell/dropout_4/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-lstm_1/while/lstm_cell/dropout_4/GreaterEqual?
%lstm_1/while/lstm_cell/dropout_4/CastCast1lstm_1/while/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%lstm_1/while/lstm_cell/dropout_4/Cast?
&lstm_1/while/lstm_cell/dropout_4/Mul_1Mul(lstm_1/while/lstm_cell/dropout_4/Mul:z:0)lstm_1/while/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&lstm_1/while/lstm_cell/dropout_4/Mul_1?
&lstm_1/while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&lstm_1/while/lstm_cell/dropout_5/Const?
$lstm_1/while/lstm_cell/dropout_5/MulMul+lstm_1/while/lstm_cell/ones_like_1:output:0/lstm_1/while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2&
$lstm_1/while/lstm_cell/dropout_5/Mul?
&lstm_1/while/lstm_cell/dropout_5/ShapeShape+lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell/dropout_5/Shape?
=lstm_1/while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=lstm_1/while/lstm_cell/dropout_5/random_uniform/RandomUniform?
/lstm_1/while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>21
/lstm_1/while/lstm_cell/dropout_5/GreaterEqual/y?
-lstm_1/while/lstm_cell/dropout_5/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-lstm_1/while/lstm_cell/dropout_5/GreaterEqual?
%lstm_1/while/lstm_cell/dropout_5/CastCast1lstm_1/while/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%lstm_1/while/lstm_cell/dropout_5/Cast?
&lstm_1/while/lstm_cell/dropout_5/Mul_1Mul(lstm_1/while/lstm_cell/dropout_5/Mul:z:0)lstm_1/while/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&lstm_1/while/lstm_cell/dropout_5/Mul_1?
&lstm_1/while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&lstm_1/while/lstm_cell/dropout_6/Const?
$lstm_1/while/lstm_cell/dropout_6/MulMul+lstm_1/while/lstm_cell/ones_like_1:output:0/lstm_1/while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2&
$lstm_1/while/lstm_cell/dropout_6/Mul?
&lstm_1/while/lstm_cell/dropout_6/ShapeShape+lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell/dropout_6/Shape?
=lstm_1/while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=lstm_1/while/lstm_cell/dropout_6/random_uniform/RandomUniform?
/lstm_1/while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>21
/lstm_1/while/lstm_cell/dropout_6/GreaterEqual/y?
-lstm_1/while/lstm_cell/dropout_6/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-lstm_1/while/lstm_cell/dropout_6/GreaterEqual?
%lstm_1/while/lstm_cell/dropout_6/CastCast1lstm_1/while/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%lstm_1/while/lstm_cell/dropout_6/Cast?
&lstm_1/while/lstm_cell/dropout_6/Mul_1Mul(lstm_1/while/lstm_cell/dropout_6/Mul:z:0)lstm_1/while/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&lstm_1/while/lstm_cell/dropout_6/Mul_1?
&lstm_1/while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&lstm_1/while/lstm_cell/dropout_7/Const?
$lstm_1/while/lstm_cell/dropout_7/MulMul+lstm_1/while/lstm_cell/ones_like_1:output:0/lstm_1/while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2&
$lstm_1/while/lstm_cell/dropout_7/Mul?
&lstm_1/while/lstm_cell/dropout_7/ShapeShape+lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell/dropout_7/Shape?
=lstm_1/while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2珰2?
=lstm_1/while/lstm_cell/dropout_7/random_uniform/RandomUniform?
/lstm_1/while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>21
/lstm_1/while/lstm_cell/dropout_7/GreaterEqual/y?
-lstm_1/while/lstm_cell/dropout_7/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-lstm_1/while/lstm_cell/dropout_7/GreaterEqual?
%lstm_1/while/lstm_cell/dropout_7/CastCast1lstm_1/while/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%lstm_1/while/lstm_cell/dropout_7/Cast?
&lstm_1/while/lstm_cell/dropout_7/Mul_1Mul(lstm_1/while/lstm_cell/dropout_7/Mul:z:0)lstm_1/while/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&lstm_1/while/lstm_cell/dropout_7/Mul_1?
lstm_1/while/lstm_cell/mulMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0(lstm_1/while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_1/while/lstm_cell/mul?
lstm_1/while/lstm_cell/mul_1Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm_1/while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_1/while/lstm_cell/mul_1?
lstm_1/while/lstm_cell/mul_2Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm_1/while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_1/while/lstm_cell/mul_2?
lstm_1/while/lstm_cell/mul_3Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm_1/while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????E2
lstm_1/while/lstm_cell/mul_3?
&lstm_1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm_1/while/lstm_cell/split/split_dim?
+lstm_1/while/lstm_cell/split/ReadVariableOpReadVariableOp6lstm_1_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	E?*
dtype02-
+lstm_1/while/lstm_cell/split/ReadVariableOp?
lstm_1/while/lstm_cell/splitSplit/lstm_1/while/lstm_cell/split/split_dim:output:03lstm_1/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm_1/while/lstm_cell/split?
lstm_1/while/lstm_cell/MatMulMatMullstm_1/while/lstm_cell/mul:z:0%lstm_1/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/MatMul?
lstm_1/while/lstm_cell/MatMul_1MatMul lstm_1/while/lstm_cell/mul_1:z:0%lstm_1/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_1?
lstm_1/while/lstm_cell/MatMul_2MatMul lstm_1/while/lstm_cell/mul_2:z:0%lstm_1/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_2?
lstm_1/while/lstm_cell/MatMul_3MatMul lstm_1/while/lstm_cell/mul_3:z:0%lstm_1/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_3?
(lstm_1/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(lstm_1/while/lstm_cell/split_1/split_dim?
-lstm_1/while/lstm_cell/split_1/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02/
-lstm_1/while/lstm_cell/split_1/ReadVariableOp?
lstm_1/while/lstm_cell/split_1Split1lstm_1/while/lstm_cell/split_1/split_dim:output:05lstm_1/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
lstm_1/while/lstm_cell/split_1?
lstm_1/while/lstm_cell/BiasAddBiasAdd'lstm_1/while/lstm_cell/MatMul:product:0'lstm_1/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
lstm_1/while/lstm_cell/BiasAdd?
 lstm_1/while/lstm_cell/BiasAdd_1BiasAdd)lstm_1/while/lstm_cell/MatMul_1:product:0'lstm_1/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 lstm_1/while/lstm_cell/BiasAdd_1?
 lstm_1/while/lstm_cell/BiasAdd_2BiasAdd)lstm_1/while/lstm_cell/MatMul_2:product:0'lstm_1/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 lstm_1/while/lstm_cell/BiasAdd_2?
 lstm_1/while/lstm_cell/BiasAdd_3BiasAdd)lstm_1/while/lstm_cell/MatMul_3:product:0'lstm_1/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 lstm_1/while/lstm_cell/BiasAdd_3?
lstm_1/while/lstm_cell/mul_4Mullstm_1_while_placeholder_3*lstm_1/while/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_4?
lstm_1/while/lstm_cell/mul_5Mullstm_1_while_placeholder_3*lstm_1/while/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_5?
lstm_1/while/lstm_cell/mul_6Mullstm_1_while_placeholder_3*lstm_1/while/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_6?
lstm_1/while/lstm_cell/mul_7Mullstm_1_while_placeholder_3*lstm_1/while/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_7?
%lstm_1/while/lstm_cell/ReadVariableOpReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02'
%lstm_1/while/lstm_cell/ReadVariableOp?
*lstm_1/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_1/while/lstm_cell/strided_slice/stack?
,lstm_1/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_1/while/lstm_cell/strided_slice/stack_1?
,lstm_1/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_1/while/lstm_cell/strided_slice/stack_2?
$lstm_1/while/lstm_cell/strided_sliceStridedSlice-lstm_1/while/lstm_cell/ReadVariableOp:value:03lstm_1/while/lstm_cell/strided_slice/stack:output:05lstm_1/while/lstm_cell/strided_slice/stack_1:output:05lstm_1/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$lstm_1/while/lstm_cell/strided_slice?
lstm_1/while/lstm_cell/MatMul_4MatMul lstm_1/while/lstm_cell/mul_4:z:0-lstm_1/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_4?
lstm_1/while/lstm_cell/addAddV2'lstm_1/while/lstm_cell/BiasAdd:output:0)lstm_1/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/add?
lstm_1/while/lstm_cell/SigmoidSigmoidlstm_1/while/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
lstm_1/while/lstm_cell/Sigmoid?
'lstm_1/while/lstm_cell/ReadVariableOp_1ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'lstm_1/while/lstm_cell/ReadVariableOp_1?
,lstm_1/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_1/while/lstm_cell/strided_slice_1/stack?
.lstm_1/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell/strided_slice_1/stack_1?
.lstm_1/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell/strided_slice_1/stack_2?
&lstm_1/while/lstm_cell/strided_slice_1StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_1:value:05lstm_1/while/lstm_cell/strided_slice_1/stack:output:07lstm_1/while/lstm_cell/strided_slice_1/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell/strided_slice_1?
lstm_1/while/lstm_cell/MatMul_5MatMul lstm_1/while/lstm_cell/mul_5:z:0/lstm_1/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_5?
lstm_1/while/lstm_cell/add_1AddV2)lstm_1/while/lstm_cell/BiasAdd_1:output:0)lstm_1/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/add_1?
 lstm_1/while/lstm_cell/Sigmoid_1Sigmoid lstm_1/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_1/while/lstm_cell/Sigmoid_1?
lstm_1/while/lstm_cell/mul_8Mul$lstm_1/while/lstm_cell/Sigmoid_1:y:0lstm_1_while_placeholder_4*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_8?
'lstm_1/while/lstm_cell/ReadVariableOp_2ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'lstm_1/while/lstm_cell/ReadVariableOp_2?
,lstm_1/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_1/while/lstm_cell/strided_slice_2/stack?
.lstm_1/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell/strided_slice_2/stack_1?
.lstm_1/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell/strided_slice_2/stack_2?
&lstm_1/while/lstm_cell/strided_slice_2StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_2:value:05lstm_1/while/lstm_cell/strided_slice_2/stack:output:07lstm_1/while/lstm_cell/strided_slice_2/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell/strided_slice_2?
lstm_1/while/lstm_cell/MatMul_6MatMul lstm_1/while/lstm_cell/mul_6:z:0/lstm_1/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_6?
lstm_1/while/lstm_cell/add_2AddV2)lstm_1/while/lstm_cell/BiasAdd_2:output:0)lstm_1/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/add_2?
lstm_1/while/lstm_cell/TanhTanh lstm_1/while/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/Tanh?
lstm_1/while/lstm_cell/mul_9Mul"lstm_1/while/lstm_cell/Sigmoid:y:0lstm_1/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_9?
lstm_1/while/lstm_cell/add_3AddV2 lstm_1/while/lstm_cell/mul_8:z:0 lstm_1/while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/add_3?
'lstm_1/while/lstm_cell/ReadVariableOp_3ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'lstm_1/while/lstm_cell/ReadVariableOp_3?
,lstm_1/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_1/while/lstm_cell/strided_slice_3/stack?
.lstm_1/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_1/while/lstm_cell/strided_slice_3/stack_1?
.lstm_1/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell/strided_slice_3/stack_2?
&lstm_1/while/lstm_cell/strided_slice_3StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_3:value:05lstm_1/while/lstm_cell/strided_slice_3/stack:output:07lstm_1/while/lstm_cell/strided_slice_3/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell/strided_slice_3?
lstm_1/while/lstm_cell/MatMul_7MatMul lstm_1/while/lstm_cell/mul_7:z:0/lstm_1/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_7?
lstm_1/while/lstm_cell/add_4AddV2)lstm_1/while/lstm_cell/BiasAdd_3:output:0)lstm_1/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/add_4?
 lstm_1/while/lstm_cell/Sigmoid_2Sigmoid lstm_1/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_1/while/lstm_cell/Sigmoid_2?
lstm_1/while/lstm_cell/Tanh_1Tanh lstm_1/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/Tanh_1?
lstm_1/while/lstm_cell/mul_10Mul$lstm_1/while/lstm_cell/Sigmoid_2:y:0!lstm_1/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_10?
lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile/multiples?
lstm_1/while/TileTile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0$lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile?
lstm_1/while/SelectV2SelectV2lstm_1/while/Tile:output:0!lstm_1/while/lstm_cell/mul_10:z:0lstm_1_while_placeholder_2*
T0*(
_output_shapes
:??????????2
lstm_1/while/SelectV2?
lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_1/multiples?
lstm_1/while/Tile_1Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_1?
lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_2/multiples?
lstm_1/while/Tile_2Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_2?
lstm_1/while/SelectV2_1SelectV2lstm_1/while/Tile_1:output:0!lstm_1/while/lstm_cell/mul_10:z:0lstm_1_while_placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_1/while/SelectV2_1?
lstm_1/while/SelectV2_2SelectV2lstm_1/while/Tile_2:output:0 lstm_1/while/lstm_cell/add_3:z:0lstm_1_while_placeholder_4*
T0*(
_output_shapes
:??????????2
lstm_1/while/SelectV2_2?
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholderlstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype023
1lstm_1/while/TensorArrayV2Write/TensorListSetItemj
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add/y?
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/addn
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add_1/y?
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/add_1?
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity?
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations^lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_1?
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_2?
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_3?
lstm_1/while/Identity_4Identitylstm_1/while/SelectV2:output:0^lstm_1/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_1/while/Identity_4?
lstm_1/while/Identity_5Identity lstm_1/while/SelectV2_1:output:0^lstm_1/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_1/while/Identity_5?
lstm_1/while/Identity_6Identity lstm_1/while/SelectV2_2:output:0^lstm_1/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_1/while/Identity_6?
lstm_1/while/NoOpNoOp&^lstm_1/while/lstm_cell/ReadVariableOp(^lstm_1/while/lstm_cell/ReadVariableOp_1(^lstm_1/while/lstm_cell/ReadVariableOp_2(^lstm_1/while/lstm_cell/ReadVariableOp_3,^lstm_1/while/lstm_cell/split/ReadVariableOp.^lstm_1/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_1/while/NoOp"7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0";
lstm_1_while_identity_6 lstm_1/while/Identity_6:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"b
.lstm_1_while_lstm_cell_readvariableop_resource0lstm_1_while_lstm_cell_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_split_1_readvariableop_resource8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0"n
4lstm_1_while_lstm_cell_split_readvariableop_resource6lstm_1_while_lstm_cell_split_readvariableop_resource_0"?
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensorelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :??????????:??????????:??????????: : : : : : 2N
%lstm_1/while/lstm_cell/ReadVariableOp%lstm_1/while/lstm_cell/ReadVariableOp2R
'lstm_1/while/lstm_cell/ReadVariableOp_1'lstm_1/while/lstm_cell/ReadVariableOp_12R
'lstm_1/while/lstm_cell/ReadVariableOp_2'lstm_1/while/lstm_cell/ReadVariableOp_22R
'lstm_1/while/lstm_cell/ReadVariableOp_3'lstm_1/while/lstm_cell/ReadVariableOp_32Z
+lstm_1/while/lstm_cell/split/ReadVariableOp+lstm_1/while/lstm_cell/split/ReadVariableOp2^
-lstm_1/while/lstm_cell/split_1/ReadVariableOp-lstm_1/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
??
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_20332
inputs_0:
'lstm_cell_split_readvariableop_resource:	E?8
)lstm_cell_split_1_readvariableop_resource:	?5
!lstm_cell_readvariableop_resource:
??
identity

identity_1

identity_2??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????E2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????E*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/ones_likex
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_1?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_cell/mul_3x
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	E?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_20196*
condR
while_cond_20195*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityn

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1n

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????E: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????E
"
_user_specified_name
inputs/0
?
?
%__inference_dense_layer_call_fn_21485

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_186052
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
lstm_1_while_body_19440*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0i
elstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0I
6lstm_1_while_lstm_cell_split_readvariableop_resource_0:	E?G
8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0:	?D
0lstm_1_while_lstm_cell_readvariableop_resource_0:
??
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5
lstm_1_while_identity_6'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorg
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensorG
4lstm_1_while_lstm_cell_split_readvariableop_resource:	E?E
6lstm_1_while_lstm_cell_split_1_readvariableop_resource:	?B
.lstm_1_while_lstm_cell_readvariableop_resource:
????%lstm_1/while/lstm_cell/ReadVariableOp?'lstm_1/while/lstm_cell/ReadVariableOp_1?'lstm_1/while/lstm_cell/ReadVariableOp_2?'lstm_1/while/lstm_cell/ReadVariableOp_3?+lstm_1/while/lstm_cell/split/ReadVariableOp?-lstm_1/while/lstm_cell/split_1/ReadVariableOp?
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????E   2@
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????E*
element_dtype022
0lstm_1/while/TensorArrayV2Read/TensorListGetItem?
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0lstm_1_while_placeholderIlstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
24
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItem?
&lstm_1/while/lstm_cell/ones_like/ShapeShape7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell/ones_like/Shape?
&lstm_1/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&lstm_1/while/lstm_cell/ones_like/Const?
 lstm_1/while/lstm_cell/ones_likeFill/lstm_1/while/lstm_cell/ones_like/Shape:output:0/lstm_1/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????E2"
 lstm_1/while/lstm_cell/ones_like?
(lstm_1/while/lstm_cell/ones_like_1/ShapeShapelstm_1_while_placeholder_3*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell/ones_like_1/Shape?
(lstm_1/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(lstm_1/while/lstm_cell/ones_like_1/Const?
"lstm_1/while/lstm_cell/ones_like_1Fill1lstm_1/while/lstm_cell/ones_like_1/Shape:output:01lstm_1/while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_1/while/lstm_cell/ones_like_1?
lstm_1/while/lstm_cell/mulMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/while/lstm_cell/mul?
lstm_1/while/lstm_cell/mul_1Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/while/lstm_cell/mul_1?
lstm_1/while/lstm_cell/mul_2Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/while/lstm_cell/mul_2?
lstm_1/while/lstm_cell/mul_3Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:?????????E2
lstm_1/while/lstm_cell/mul_3?
&lstm_1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm_1/while/lstm_cell/split/split_dim?
+lstm_1/while/lstm_cell/split/ReadVariableOpReadVariableOp6lstm_1_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	E?*
dtype02-
+lstm_1/while/lstm_cell/split/ReadVariableOp?
lstm_1/while/lstm_cell/splitSplit/lstm_1/while/lstm_cell/split/split_dim:output:03lstm_1/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	E?:	E?:	E?:	E?*
	num_split2
lstm_1/while/lstm_cell/split?
lstm_1/while/lstm_cell/MatMulMatMullstm_1/while/lstm_cell/mul:z:0%lstm_1/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/MatMul?
lstm_1/while/lstm_cell/MatMul_1MatMul lstm_1/while/lstm_cell/mul_1:z:0%lstm_1/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_1?
lstm_1/while/lstm_cell/MatMul_2MatMul lstm_1/while/lstm_cell/mul_2:z:0%lstm_1/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_2?
lstm_1/while/lstm_cell/MatMul_3MatMul lstm_1/while/lstm_cell/mul_3:z:0%lstm_1/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_3?
(lstm_1/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(lstm_1/while/lstm_cell/split_1/split_dim?
-lstm_1/while/lstm_cell/split_1/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02/
-lstm_1/while/lstm_cell/split_1/ReadVariableOp?
lstm_1/while/lstm_cell/split_1Split1lstm_1/while/lstm_cell/split_1/split_dim:output:05lstm_1/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
lstm_1/while/lstm_cell/split_1?
lstm_1/while/lstm_cell/BiasAddBiasAdd'lstm_1/while/lstm_cell/MatMul:product:0'lstm_1/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
lstm_1/while/lstm_cell/BiasAdd?
 lstm_1/while/lstm_cell/BiasAdd_1BiasAdd)lstm_1/while/lstm_cell/MatMul_1:product:0'lstm_1/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 lstm_1/while/lstm_cell/BiasAdd_1?
 lstm_1/while/lstm_cell/BiasAdd_2BiasAdd)lstm_1/while/lstm_cell/MatMul_2:product:0'lstm_1/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 lstm_1/while/lstm_cell/BiasAdd_2?
 lstm_1/while/lstm_cell/BiasAdd_3BiasAdd)lstm_1/while/lstm_cell/MatMul_3:product:0'lstm_1/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 lstm_1/while/lstm_cell/BiasAdd_3?
lstm_1/while/lstm_cell/mul_4Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_4?
lstm_1/while/lstm_cell/mul_5Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_5?
lstm_1/while/lstm_cell/mul_6Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_6?
lstm_1/while/lstm_cell/mul_7Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_7?
%lstm_1/while/lstm_cell/ReadVariableOpReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02'
%lstm_1/while/lstm_cell/ReadVariableOp?
*lstm_1/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_1/while/lstm_cell/strided_slice/stack?
,lstm_1/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_1/while/lstm_cell/strided_slice/stack_1?
,lstm_1/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm_1/while/lstm_cell/strided_slice/stack_2?
$lstm_1/while/lstm_cell/strided_sliceStridedSlice-lstm_1/while/lstm_cell/ReadVariableOp:value:03lstm_1/while/lstm_cell/strided_slice/stack:output:05lstm_1/while/lstm_cell/strided_slice/stack_1:output:05lstm_1/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$lstm_1/while/lstm_cell/strided_slice?
lstm_1/while/lstm_cell/MatMul_4MatMul lstm_1/while/lstm_cell/mul_4:z:0-lstm_1/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_4?
lstm_1/while/lstm_cell/addAddV2'lstm_1/while/lstm_cell/BiasAdd:output:0)lstm_1/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/add?
lstm_1/while/lstm_cell/SigmoidSigmoidlstm_1/while/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
lstm_1/while/lstm_cell/Sigmoid?
'lstm_1/while/lstm_cell/ReadVariableOp_1ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'lstm_1/while/lstm_cell/ReadVariableOp_1?
,lstm_1/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_1/while/lstm_cell/strided_slice_1/stack?
.lstm_1/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell/strided_slice_1/stack_1?
.lstm_1/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell/strided_slice_1/stack_2?
&lstm_1/while/lstm_cell/strided_slice_1StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_1:value:05lstm_1/while/lstm_cell/strided_slice_1/stack:output:07lstm_1/while/lstm_cell/strided_slice_1/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell/strided_slice_1?
lstm_1/while/lstm_cell/MatMul_5MatMul lstm_1/while/lstm_cell/mul_5:z:0/lstm_1/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_5?
lstm_1/while/lstm_cell/add_1AddV2)lstm_1/while/lstm_cell/BiasAdd_1:output:0)lstm_1/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/add_1?
 lstm_1/while/lstm_cell/Sigmoid_1Sigmoid lstm_1/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_1/while/lstm_cell/Sigmoid_1?
lstm_1/while/lstm_cell/mul_8Mul$lstm_1/while/lstm_cell/Sigmoid_1:y:0lstm_1_while_placeholder_4*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_8?
'lstm_1/while/lstm_cell/ReadVariableOp_2ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'lstm_1/while/lstm_cell/ReadVariableOp_2?
,lstm_1/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_1/while/lstm_cell/strided_slice_2/stack?
.lstm_1/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell/strided_slice_2/stack_1?
.lstm_1/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell/strided_slice_2/stack_2?
&lstm_1/while/lstm_cell/strided_slice_2StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_2:value:05lstm_1/while/lstm_cell/strided_slice_2/stack:output:07lstm_1/while/lstm_cell/strided_slice_2/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell/strided_slice_2?
lstm_1/while/lstm_cell/MatMul_6MatMul lstm_1/while/lstm_cell/mul_6:z:0/lstm_1/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_6?
lstm_1/while/lstm_cell/add_2AddV2)lstm_1/while/lstm_cell/BiasAdd_2:output:0)lstm_1/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/add_2?
lstm_1/while/lstm_cell/TanhTanh lstm_1/while/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/Tanh?
lstm_1/while/lstm_cell/mul_9Mul"lstm_1/while/lstm_cell/Sigmoid:y:0lstm_1/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_9?
lstm_1/while/lstm_cell/add_3AddV2 lstm_1/while/lstm_cell/mul_8:z:0 lstm_1/while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/add_3?
'lstm_1/while/lstm_cell/ReadVariableOp_3ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'lstm_1/while/lstm_cell/ReadVariableOp_3?
,lstm_1/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_1/while/lstm_cell/strided_slice_3/stack?
.lstm_1/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm_1/while/lstm_cell/strided_slice_3/stack_1?
.lstm_1/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell/strided_slice_3/stack_2?
&lstm_1/while/lstm_cell/strided_slice_3StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_3:value:05lstm_1/while/lstm_cell/strided_slice_3/stack:output:07lstm_1/while/lstm_cell/strided_slice_3/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell/strided_slice_3?
lstm_1/while/lstm_cell/MatMul_7MatMul lstm_1/while/lstm_cell/mul_7:z:0/lstm_1/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2!
lstm_1/while/lstm_cell/MatMul_7?
lstm_1/while/lstm_cell/add_4AddV2)lstm_1/while/lstm_cell/BiasAdd_3:output:0)lstm_1/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/add_4?
 lstm_1/while/lstm_cell/Sigmoid_2Sigmoid lstm_1/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_1/while/lstm_cell/Sigmoid_2?
lstm_1/while/lstm_cell/Tanh_1Tanh lstm_1/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/Tanh_1?
lstm_1/while/lstm_cell/mul_10Mul$lstm_1/while/lstm_cell/Sigmoid_2:y:0!lstm_1/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_1/while/lstm_cell/mul_10?
lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile/multiples?
lstm_1/while/TileTile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0$lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile?
lstm_1/while/SelectV2SelectV2lstm_1/while/Tile:output:0!lstm_1/while/lstm_cell/mul_10:z:0lstm_1_while_placeholder_2*
T0*(
_output_shapes
:??????????2
lstm_1/while/SelectV2?
lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_1/multiples?
lstm_1/while/Tile_1Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_1?
lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_2/multiples?
lstm_1/while/Tile_2Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_2?
lstm_1/while/SelectV2_1SelectV2lstm_1/while/Tile_1:output:0!lstm_1/while/lstm_cell/mul_10:z:0lstm_1_while_placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_1/while/SelectV2_1?
lstm_1/while/SelectV2_2SelectV2lstm_1/while/Tile_2:output:0 lstm_1/while/lstm_cell/add_3:z:0lstm_1_while_placeholder_4*
T0*(
_output_shapes
:??????????2
lstm_1/while/SelectV2_2?
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholderlstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype023
1lstm_1/while/TensorArrayV2Write/TensorListSetItemj
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add/y?
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/addn
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add_1/y?
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/add_1?
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity?
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations^lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_1?
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_2?
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_3?
lstm_1/while/Identity_4Identitylstm_1/while/SelectV2:output:0^lstm_1/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_1/while/Identity_4?
lstm_1/while/Identity_5Identity lstm_1/while/SelectV2_1:output:0^lstm_1/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_1/while/Identity_5?
lstm_1/while/Identity_6Identity lstm_1/while/SelectV2_2:output:0^lstm_1/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_1/while/Identity_6?
lstm_1/while/NoOpNoOp&^lstm_1/while/lstm_cell/ReadVariableOp(^lstm_1/while/lstm_cell/ReadVariableOp_1(^lstm_1/while/lstm_cell/ReadVariableOp_2(^lstm_1/while/lstm_cell/ReadVariableOp_3,^lstm_1/while/lstm_cell/split/ReadVariableOp.^lstm_1/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_1/while/NoOp"7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0";
lstm_1_while_identity_6 lstm_1/while/Identity_6:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"b
.lstm_1_while_lstm_cell_readvariableop_resource0lstm_1_while_lstm_cell_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_split_1_readvariableop_resource8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0"n
4lstm_1_while_lstm_cell_split_readvariableop_resource6lstm_1_while_lstm_cell_split_readvariableop_resource_0"?
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensorelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :??????????:??????????:??????????: : : : : : 2N
%lstm_1/while/lstm_cell/ReadVariableOp%lstm_1/while/lstm_cell/ReadVariableOp2R
'lstm_1/while/lstm_cell/ReadVariableOp_1'lstm_1/while/lstm_cell/ReadVariableOp_12R
'lstm_1/while/lstm_cell/ReadVariableOp_2'lstm_1/while/lstm_cell/ReadVariableOp_22R
'lstm_1/while/lstm_cell/ReadVariableOp_3'lstm_1/while/lstm_cell/ReadVariableOp_32Z
+lstm_1/while/lstm_cell/split/ReadVariableOp+lstm_1/while/lstm_cell/split/ReadVariableOp2^
-lstm_1/while/lstm_cell/split_1/ReadVariableOp-lstm_1/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
?
'__inference_dense_1_layer_call_fn_21504

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_186282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_21531

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_186822
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
?__inference_lstm_layer_call_and_return_conditional_losses_18635
x
lstm_1_18583:	E?
lstm_1_18585:	? 
lstm_1_18587:
??
dense_18606:
??
dense_18608:	? 
dense_1_18629:	?
dense_1_18631:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?lstm_1/StatefulPartitionedCallO
x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
x_1?
NotEqualNotEqualxx_1:output:0*
T0*+
_output_shapes
:?????????E*
incompatible_shape_error( 2

NotEqualp
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Any/reduction_indicesh
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*'
_output_shapes
:?????????2
Any?
lstm_1/StatefulPartitionedCallStatefulPartitionedCallxAny:output:0lstm_1_18583lstm_1_18585lstm_1_18587*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_185822 
lstm_1/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2'lstm_1/StatefulPartitionedCall:output:0'lstm_1/StatefulPartitionedCall:output:1'lstm_1/StatefulPartitionedCall:output:2concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_18606dense_18608*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_186052
dense/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_186162
dropout/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_18629dense_1_18631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_186282!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????E: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:N J
+
_output_shapes
:?????????E

_user_specified_namex
?

?
while_cond_18904
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_18904___redundant_placeholder03
/while_while_cond_18904___redundant_placeholder13
/while_while_cond_18904___redundant_placeholder23
/while_while_cond_18904___redundant_placeholder33
/while_while_cond_18904___redundant_placeholder4
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :??????????:??????????:??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_21509

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????E<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
lstm

dense1

dense2
dropout
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
*f&call_and_return_all_conditional_losses
g_default_save_signature
h__call__"
_tf_keras_model
?
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
*i&call_and_return_all_conditional_losses
j__call__"
_tf_keras_rnn_layer
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*k&call_and_return_all_conditional_losses
l__call__"
_tf_keras_layer
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*m&call_and_return_all_conditional_losses
n__call__"
_tf_keras_layer
?
trainable_variables
regularization_losses
	variables
 	keras_api
*o&call_and_return_all_conditional_losses
p__call__"
_tf_keras_layer
?
!iter

"beta_1

#beta_2
	$decay
%learning_ratemXmYmZm[&m\'m](m^v_v`vavb&vc'vd(ve"
	optimizer
Q
&0
'1
(2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
&0
'1
(2
3
4
5
6"
trackable_list_wrapper
?
)layer_regularization_losses
*metrics
trainable_variables
+non_trainable_variables
,layer_metrics
regularization_losses
	variables

-layers
h__call__
g_default_save_signature
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
,
qserving_default"
signature_map
?
.
state_size

&kernel
'recurrent_kernel
(bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
*r&call_and_return_all_conditional_losses
s__call__"
_tf_keras_layer
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
?
3layer_regularization_losses
4metrics
trainable_variables

5states
6non_trainable_variables
7layer_metrics
regularization_losses
	variables

8layers
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
%:#
??2lstm/dense/kernel
:?2lstm/dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
9layer_regularization_losses

:layers
;metrics
trainable_variables
<non_trainable_variables
regularization_losses
	variables
=layer_metrics
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
&:$	?2lstm/dense_1/kernel
:2lstm/dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
>layer_regularization_losses

?layers
@metrics
trainable_variables
Anon_trainable_variables
regularization_losses
	variables
Blayer_metrics
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Clayer_regularization_losses

Dlayers
Emetrics
trainable_variables
Fnon_trainable_variables
regularization_losses
	variables
Glayer_metrics
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
/:-	E?2lstm/lstm_1/lstm_cell/kernel
::8
??2&lstm/lstm_1/lstm_cell/recurrent_kernel
):'?2lstm/lstm_1/lstm_cell/bias
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
?
Jlayer_regularization_losses

Klayers
Lmetrics
/trainable_variables
Mnon_trainable_variables
0regularization_losses
1	variables
Nlayer_metrics
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
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
'
0"
trackable_list_wrapper
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
N
	Ototal
	Pcount
Q	variables
R	keras_api"
_tf_keras_metric
^
	Stotal
	Tcount
U
_fn_kwargs
V	variables
W	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
.
O0
P1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
S0
T1"
trackable_list_wrapper
-
V	variables"
_generic_user_object
*:(
??2Adam/lstm/dense/kernel/m
#:!?2Adam/lstm/dense/bias/m
+:)	?2Adam/lstm/dense_1/kernel/m
$:"2Adam/lstm/dense_1/bias/m
4:2	E?2#Adam/lstm/lstm_1/lstm_cell/kernel/m
?:=
??2-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/m
.:,?2!Adam/lstm/lstm_1/lstm_cell/bias/m
*:(
??2Adam/lstm/dense/kernel/v
#:!?2Adam/lstm/dense/bias/v
+:)	?2Adam/lstm/dense_1/kernel/v
$:"2Adam/lstm/dense_1/bias/v
4:2	E?2#Adam/lstm/lstm_1/lstm_cell/kernel/v
?:=
??2-Adam/lstm/lstm_1/lstm_cell/recurrent_kernel/v
.:,?2!Adam/lstm/lstm_1/lstm_cell/bias/v
?2?
?__inference_lstm_layer_call_and_return_conditional_losses_19608
?__inference_lstm_layer_call_and_return_conditional_losses_20041
?__inference_lstm_layer_call_and_return_conditional_losses_19253
?__inference_lstm_layer_call_and_return_conditional_losses_19283?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
 __inference__wrapped_model_17473input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_lstm_layer_call_fn_18652
$__inference_lstm_layer_call_fn_20060
$__inference_lstm_layer_call_fn_20079
$__inference_lstm_layer_call_fn_19223?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_lstm_1_layer_call_and_return_conditional_losses_20332
A__inference_lstm_1_layer_call_and_return_conditional_losses_20713
A__inference_lstm_1_layer_call_and_return_conditional_losses_20994
A__inference_lstm_1_layer_call_and_return_conditional_losses_21403?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lstm_1_layer_call_fn_21418
&__inference_lstm_1_layer_call_fn_21433
&__inference_lstm_1_layer_call_fn_21449
&__inference_lstm_1_layer_call_fn_21465?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_21476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_21485?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_21495?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_21504?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_21509
B__inference_dropout_layer_call_and_return_conditional_losses_21521?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_21526
'__inference_dropout_layer_call_fn_21531?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_19310input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_21613
D__inference_lstm_cell_layer_call_and_return_conditional_losses_21759?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_lstm_cell_layer_call_fn_21776
)__inference_lstm_cell_layer_call_fn_21793?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_17473t&('4?1
*?'
%?"
input_1?????????E
? "3?0
.
output_1"?
output_1??????????
B__inference_dense_1_layer_call_and_return_conditional_losses_21495]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_1_layer_call_fn_21504P0?-
&?#
!?
inputs??????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_21476^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
%__inference_dense_layer_call_fn_21485Q0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dropout_layer_call_and_return_conditional_losses_21509^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_21521^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? |
'__inference_dropout_layer_call_fn_21526Q4?1
*?'
!?
inputs??????????
p 
? "???????????|
'__inference_dropout_layer_call_fn_21531Q4?1
*?'
!?
inputs??????????
p
? "????????????
A__inference_lstm_1_layer_call_and_return_conditional_losses_20332?&('O?L
E?B
4?1
/?,
inputs/0??????????????????E

 
p 

 
? "m?j
c?`
?
0/0??????????
?
0/1??????????
?
0/2??????????
? ?
A__inference_lstm_1_layer_call_and_return_conditional_losses_20713?&('O?L
E?B
4?1
/?,
inputs/0??????????????????E

 
p

 
? "m?j
c?`
?
0/0??????????
?
0/1??????????
?
0/2??????????
? ?
A__inference_lstm_1_layer_call_and_return_conditional_losses_20994?&('[?X
Q?N
$?!
inputs?????????E
?
mask?????????

p 

 
? "m?j
c?`
?
0/0??????????
?
0/1??????????
?
0/2??????????
? ?
A__inference_lstm_1_layer_call_and_return_conditional_losses_21403?&('[?X
Q?N
$?!
inputs?????????E
?
mask?????????

p

 
? "m?j
c?`
?
0/0??????????
?
0/1??????????
?
0/2??????????
? ?
&__inference_lstm_1_layer_call_fn_21418?&('O?L
E?B
4?1
/?,
inputs/0??????????????????E

 
p 

 
? "]?Z
?
0??????????
?
1??????????
?
2???????????
&__inference_lstm_1_layer_call_fn_21433?&('O?L
E?B
4?1
/?,
inputs/0??????????????????E

 
p

 
? "]?Z
?
0??????????
?
1??????????
?
2???????????
&__inference_lstm_1_layer_call_fn_21449?&('[?X
Q?N
$?!
inputs?????????E
?
mask?????????

p 

 
? "]?Z
?
0??????????
?
1??????????
?
2???????????
&__inference_lstm_1_layer_call_fn_21465?&('[?X
Q?N
$?!
inputs?????????E
?
mask?????????

p

 
? "]?Z
?
0??????????
?
1??????????
?
2???????????
D__inference_lstm_cell_layer_call_and_return_conditional_losses_21613?&('??
x?u
 ?
inputs?????????E
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_21759?&('??
x?u
 ?
inputs?????????E
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
)__inference_lstm_cell_layer_call_fn_21776?&('??
x?u
 ?
inputs?????????E
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
)__inference_lstm_cell_layer_call_fn_21793?&('??
x?u
 ?
inputs?????????E
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
?__inference_lstm_layer_call_and_return_conditional_losses_19253j&('8?5
.?+
%?"
input_1?????????E
p 
? "%?"
?
0?????????
? ?
?__inference_lstm_layer_call_and_return_conditional_losses_19283j&('8?5
.?+
%?"
input_1?????????E
p
? "%?"
?
0?????????
? ?
?__inference_lstm_layer_call_and_return_conditional_losses_19608d&('2?/
(?%
?
x?????????E
p 
? "%?"
?
0?????????
? ?
?__inference_lstm_layer_call_and_return_conditional_losses_20041d&('2?/
(?%
?
x?????????E
p
? "%?"
?
0?????????
? ?
$__inference_lstm_layer_call_fn_18652]&('8?5
.?+
%?"
input_1?????????E
p 
? "???????????
$__inference_lstm_layer_call_fn_19223]&('8?5
.?+
%?"
input_1?????????E
p
? "??????????
$__inference_lstm_layer_call_fn_20060W&('2?/
(?%
?
x?????????E
p 
? "??????????
$__inference_lstm_layer_call_fn_20079W&('2?/
(?%
?
x?????????E
p
? "???????????
#__inference_signature_wrapper_19310&('??<
? 
5?2
0
input_1%?"
input_1?????????E"3?0
.
output_1"?
output_1?????????