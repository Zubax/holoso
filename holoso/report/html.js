// Interactive layer for the schedule grid. _sched_script in html.py substitutes the per-module payload into the
// placeholder on the "var data =" line below. Kept readable (the report is a tool, not a minified asset); without JS
// the grid still renders fully -- only the dataflow edges and hover behaviors are absent.
(function () {
    "use strict";
    var data = __DATA__;
    var edges = data.edges;          // [writeCellId, operandCellId, color, operationGroup]
    var columns = data.columns;      // label per grid column, indexed by (cell index - 1)
    var constants = data.constants;  // { "c0": "1.0", ... }
    var liveness = data.liveness;    // { "<registerIndex>": [[start, end], ...] } live-row intervals

    var wrap = document.getElementById("schedwrap");
    if (!wrap) {
        return;
    }
    var svg = wrap.querySelector("svg.edges");
    var SVG_NS = "http://www.w3.org/2000/svg";

    // Elements grouped by operation, so a hover can light up just one operation's nodes. The result cells and the ops
    // chip are static (tagged with data-op); the edges are rebuilt on every redraw and tracked separately.
    var nodesByGroup = {};
    wrap.querySelectorAll("[data-op]").forEach(function (element) {
        var group = element.dataset.op;
        (nodesByGroup[group] = nodesByGroup[group] || []).push(element);
    });
    var edgesByGroup = {};

    // Draw each dataflow edge (result cell -> operand cell) by measuring the rendered cell centres, so the overlay
    // stays aligned no matter the exact cell metrics or a late web-font swap.
    function drawEdges() {
        edgesByGroup = {};
        while (svg.firstChild) {
            svg.removeChild(svg.firstChild);
        }
        var origin = wrap.getBoundingClientRect();
        svg.setAttribute("width", wrap.scrollWidth);
        svg.setAttribute("height", wrap.scrollHeight);
        edges.forEach(function (edge) {
            var fromCell = document.getElementById(edge[0]);
            var toCell = document.getElementById(edge[1]);
            if (!fromCell || !toCell || fromCell === toCell) {
                return;
            }
            var from = fromCell.getBoundingClientRect();
            var to = toCell.getBoundingClientRect();
            var x1 = from.left - origin.left + from.width / 2;
            var y1 = from.top - origin.top + from.height / 2;
            var x2 = to.left - origin.left + to.width / 2;
            var y2 = to.top - origin.top + to.height / 2;
            var color = edge[2];
            var group = edge[3];

            var line = document.createElementNS(SVG_NS, "line");
            line.setAttribute("x1", x1);
            line.setAttribute("y1", y1);
            line.setAttribute("x2", x2);
            line.setAttribute("y2", y2);
            line.setAttribute("stroke", color);
            line.setAttribute("stroke-width", "1");
            line.setAttribute("stroke-opacity", "0.85");
            svg.appendChild(line);

            var dot = document.createElementNS(SVG_NS, "circle");
            dot.setAttribute("cx", x2);
            dot.setAttribute("cy", y2);
            dot.setAttribute("r", "1.7");
            dot.setAttribute("fill", color);
            svg.appendChild(dot);

            (edgesByGroup[group] = edgesByGroup[group] || []).push(line, dot);
        });
    }

    drawEdges();
    window.addEventListener("resize", drawEdges);
    if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(drawEdges);
    }

    // Hovering an operation (its result column or its ops chip) makes only that one operation stand out: we toggle a
    // class on its own handful of elements rather than restyling every other operation, so there is no per-hover
    // sweep of the grid. The .hl class blackens its edges, result cells and chip.
    var focused = null;  // currently focused group (a "data-op" string), or null

    function setHighlighted(group, on) {
        [nodesByGroup[group], edgesByGroup[group]].forEach(function (list) {
            if (list) {
                list.forEach(function (element) {
                    element.classList.toggle("hl", on);
                });
            }
        });
    }

    function focus(group) {
        if (group === focused) {
            return;
        }
        if (focused !== null) {
            setHighlighted(focused, false);
        }
        focused = group;
        if (group !== null) {
            setHighlighted(group, true);
        }
    }

    // Whether register `label` (e.g. "r41") holds a live value on `cycle`, from its residence intervals.
    function isAlive(label, cycle) {
        var intervals = liveness[label.slice(1)];
        if (!intervals) {
            return false;
        }
        for (var i = 0; i < intervals.length; i++) {
            if (cycle >= intervals[i][0] && cycle <= intervals[i][1]) {
                return true;
            }
        }
        return false;
    }

    wrap.addEventListener("mouseover", function (event) {
        var owner = event.target.closest("[data-op]");  // a result cell or an ops chip
        focus(owner ? owner.dataset.op : null);

        var cell = event.target.closest("td");
        if (!cell || !cell.classList.contains("gc")) {
            return;
        }
        var label = columns[cell.cellIndex - 1];
        if (label === undefined) {
            return;
        }
        var clk = cell.parentNode.cells[0].textContent.trim();
        if (label.charAt(0) === "c") {
            cell.title = label + " = " + constants[label];
        } else {
            var cycle = clk === "in" ? 0 : parseInt(clk, 10);
            cell.title = label + "@" + clk + " " + (isAlive(label, cycle) ? "alive" : "dead");
        }
    });
    wrap.addEventListener("mouseout", function (event) {
        if (!wrap.contains(event.relatedTarget)) {
            focus(null);
        }
    });
})();
