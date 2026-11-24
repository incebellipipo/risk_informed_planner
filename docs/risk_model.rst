Risk computation
================

Evidence at a space-time point
------------------------------

:class:`~risk_informed_planner.risk.dynamic_risk.DynamicRisk` clears the BBN's
previous evidence before evaluating each node. Its inputs include:

* Distance to shore and depth at the node.
* Grounding-course evidence from the forward lookahead. For compatibility with
  the original study, any depth greater than the grounding hard limit on this
  segment produces YES evidence; a free segment produces NO.
* Traffic intensity from the map's traffic layer.
* Relative speed, distance at closest point of approach (DCPA), and time to
  closest point of approach (TCPA) for moving vessels.

A sampled node carries its incoming heading through ``parent`` and its distance
from the root through ``distance_cost``. At constant desired speed :math:`v`,
the target vessel is advanced to :math:`t = \mathrm{distance\_cost}/v` before
encounter geometry is calculated. With non-positive desired speed the current
implementation evaluates the target at time zero.

Only ``LinearDynamicObstacle`` objects participate in collision-course evidence.
An encounter is considered when DCPA is below its configured maximum and
:math:`0 < \mathrm{TCPA} < \mathrm{TCPA}_{\max}`. If several encounters qualify,
the selected encounter has smallest DCPA, then smallest TCPA, then largest
relative speed. With none, encounter evidence is ``Unlikely`` and the other
encounter inputs are left unset.

Continuous observations are discretized into equally spaced BBN bins, except
where an explicit interval list is used. Values beyond the range stay in an end
bin. Boundary values enter the next bin. Traffic reverses this ordering.

Normalized expected loss
------------------------

The model reads the marginal grounding and collision severity distributions
from ``BBN_autonomous_ranked.xdsl``. Severity outcomes are ordered as negligible,
marginal, moderate, critical, and catastrophic. Default weights are:

.. code-block:: python

   [0.0, 4.0, 16.0, 64.0, 256.0]

For hazard :math:`h`, let :math:`p_{h,s}` be its marginal severity probabilities
and :math:`w_s` the weights. The normalized hazard loss is

.. math::

   r_h = \frac{\sum_s p_{h,s} w_s}{\max_s w_s}.

The combined scalar uses a probabilistic OR:

.. math::

   r = 1 - \prod_h (1-r_h).

For valid probability distributions and nonnegative weights beginning at zero,
this lies in :math:`[0,1]`. The severity marginal already includes the accident
probability; the cost model does not multiply by that probability again.
The combined value is a severity-weighted index rather than the probability
of either accident.

Custom severity weights must be non-decreasing and have a positive maximum.
A nonzero negligible weight produces a warning because it adds background cost
throughout the map. See :doc:`api/risk_informed_planner/risk/cost_model/index` for the customization interface.

Edge risk and cost
------------------

Each candidate edge of length :math:`L` is evaluated at evenly spaced samples.
The edge endpoint is always included. The start point is not sampled again.
With spacing :math:`s`, the number of samples is

.. math::

   n = \min\left(\max(\lceil L/s \rceil,1), n_{\max}\right).

Unspecified or non-positive spacing uses one endpoint sample. The configured
aggregation is the mean, maximum, or endpoint risk. ``peak_risk`` always records
the largest sampled risk, regardless of the aggregation.

If :math:`\bar r` is the aggregate risk divided by the cost model's worst-case
cost, the edge cost is

.. math::

   c_e = \alpha L + (1-\alpha)\beta\bar r L
         + \gamma\frac{|\Delta\psi|}{\pi}L_{\mathrm{ref}}.

The cost-to-come adds the parent's accumulated cost. The turn reference length
defaults to ``STEP_SIZE``. The root has no incoming heading and incurs no turn
penalty. A turn exceeding ``MAX_TURN_ANGLE`` is rejected before BBN evaluation.

``RISK_CUTOFF`` gates the sampled peak, not the mean. A cheap edge can still be
rejected because of a hazardous interior sample. Finite sample spacing can miss
hazards between samples; reducing spacing increases the inference workload.

Snapshots and telemetry
-----------------------

With ``record_bbn_snapshots=True``, each evaluated edge endpoint can retain the
network's state probabilities for plotting. The stored snapshot describes the
endpoint evidence, while ``node.risk`` can describe the whole edge's aggregate.
Do not interpret these two quantities as interchangeable.

The planner exposes ``no_of_cost_evaluations``, ``risk_evaluations``, and
``risk_compute_times``. The latter measures the full ``compute_risk`` call,
including evidence construction, BBN inference, and cost computation.
