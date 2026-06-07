// Interactive layer for the schedule grid. _sched_script in html.py substitutes the per-module payload into the
// placeholder on the "var data =" line below. Kept readable (the report is a tool, not a minified asset); without JS
// the grid still renders fully -- only the dataflow edges and hover behaviors are absent.
(function () {
    "use strict";
    var data = __DATA__;
    var edges = data.edges;          // [writeCellId, operandCellId, color, operationGroup]
    var columns = data.columns;      // label per grid column, indexed by (cell index - 1)
    var constants = data.constants;  // { "c0": "1.0", ... }
    var liveness = data.liveness;    // { "<columnLabel>": [[start, end], ...] } live-row intervals, keyed by full label
    var arrows = data.arrows;        // [{ from: <srcRowCyc>, to: <dstRowCyc>, tip: <condition or "jump"> }] margin jumps

    var wrap = document.getElementById("schedwrap");
    if (!wrap) {
        return;
    }
    var svg = wrap.querySelector("svg.edges");
    var grid = wrap.querySelector("table.grid");
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
            // "state" is a sentinel for a state writeback; resolve it to the themed --c-state (color stays in CSS).
            var color = edge[2] === "state"
                ? getComputedStyle(document.documentElement).getPropertyValue("--c-state").trim()
                : edge[2];
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

    // Control-transfer arrows in the right margin: one channel (column) per non-fall-through jump, side by side so they
    // never overlap. Each routes out of the grid at its source row, right into its own channel, vertically to the target
    // row, then back left to the grid with an arrowhead. The margin is reserved as right padding on the wrapper so the
    // grid's own scrollWidth covers the channels (and the overlay, sized to it, can paint them).
    var CHANNEL_GAP = 9;    // clearance between the grid's right edge and the first channel, in px
    var CHANNEL_STEP = 11;  // spacing between adjacent channels, in px
    var HEAD = 5;           // arrowhead half-extent, in px

    function rowCentreY(cyc, origin) {
        var cell = document.getElementById("pc_" + cyc);
        if (!cell) {
            return null;
        }
        var rect = cell.getBoundingClientRect();
        return rect.top - origin.top + rect.height / 2;
    }

    function drawArrows() {
        if (!arrows.length) {
            return;
        }
        var origin = wrap.getBoundingClientRect();
        var gridRight = grid.getBoundingClientRect().right - origin.left;
        // Reserve the channel band as wrapper padding, then resize the overlay to the grown scrollWidth so it can paint
        // the full band (drawEdges sized it to the pre-padding width).
        var band = CHANNEL_GAP + arrows.length * CHANNEL_STEP + HEAD;
        wrap.style.paddingRight = band + "px";
        svg.setAttribute("width", wrap.scrollWidth);
        svg.setAttribute("height", wrap.scrollHeight);
        arrows.forEach(function (arrow, i) {
            var y1 = rowCentreY(arrow.from, origin);
            var y2 = rowCentreY(arrow.to, origin);
            if (y1 === null || y2 === null) {
                return;  // a row outside the rendered range -- skip the arrow safely
            }
            var channelX = gridRight + CHANNEL_GAP + i * CHANNEL_STEP;
            var group = document.createElementNS(SVG_NS, "g");
            group.setAttribute("class", "jarrow");

            var bracket = document.createElementNS(SVG_NS, "polyline");
            bracket.setAttribute("points",
                gridRight + "," + y1 + " " + channelX + "," + y1 + " " +
                channelX + "," + y2 + " " + gridRight + "," + y2);
            bracket.setAttribute("fill", "none");
            group.appendChild(bracket);

            var head = document.createElementNS(SVG_NS, "polygon");  // arrowhead pointing left into the target row
            head.setAttribute("points",
                gridRight + "," + y2 + " " + (gridRight + HEAD) + "," + (y2 - HEAD) + " " +
                (gridRight + HEAD) + "," + (y2 + HEAD));
            group.appendChild(head);

            // Rarefied dotted feed from the boolean register the branch tests (its cell at the source row) to the
            // arrow's root, so that register's residence ends visibly at the branch rather than in nothingness. The
            // tested cell is marked with a circle, exactly as a dataflow edge marks an operand cell.
            if (arrow.cond) {
                var condCell = document.getElementById(arrow.cond);
                if (condCell) {
                    var cr = condCell.getBoundingClientRect();
                    var cx = cr.left - origin.left + cr.width / 2;
                    var cy = cr.top - origin.top + cr.height / 2;
                    var feed = document.createElementNS(SVG_NS, "line");
                    feed.setAttribute("x1", cx);
                    feed.setAttribute("y1", cy);
                    feed.setAttribute("x2", gridRight);
                    feed.setAttribute("y2", y1);
                    feed.setAttribute("class", "jcond");
                    group.appendChild(feed);
                    var dot = document.createElementNS(SVG_NS, "circle");  // operand marker on the tested register cell
                    dot.setAttribute("cx", cx);
                    dot.setAttribute("cy", cy);
                    dot.setAttribute("r", "1.8");
                    dot.setAttribute("class", "jcond");
                    group.appendChild(dot);
                }
            }

            var title = document.createElementNS(SVG_NS, "title");
            title.textContent = arrow.tip;
            group.appendChild(title);
            svg.appendChild(group);
        });
    }

    function redraw() {
        wrap.style.paddingRight = "0px";  // release any reserved band so the grid metrics are measured bare
        drawEdges();
        drawArrows();
    }

    redraw();
    window.addEventListener("resize", redraw);
    if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(redraw);
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

    // Whether register `label` (e.g. "f41" or "b0") holds a live value on `cycle`, from its residence intervals. The
    // map is keyed by the full label, so the float and boolean banks never collide on a shared bank index.
    function isAlive(label, cycle) {
        var intervals = liveness[label];
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

    // Column crosshair: hovering any cell highlights its whole column (the row is highlighted by CSS :hover), by
    // toggling .colhl on that column's body cells. The active column is tracked so re-firing inside a cell is a no-op.
    var columnCells = [];
    var columnIndex = -1;

    function highlightColumn(index) {
        if (index === columnIndex) {
            return;
        }
        columnCells.forEach(function (cell) {
            cell.classList.remove("colhl");
        });
        columnCells = [];
        columnIndex = index;
        if (index < 0) {
            return;
        }
        var rows = grid.rows;
        for (var i = 0; i < rows.length; i++) {
            var cell = rows[i].cells[index];
            if (cell && cell.tagName === "TD") {  // body cells only; header cells span columns and are skipped
                cell.classList.add("colhl");
                columnCells.push(cell);
            }
        }
    }

    wrap.addEventListener("mouseover", function (event) {
        var owner = event.target.closest("[data-op]");  // a result cell or an ops chip
        focus(owner ? owner.dataset.op : null);

        var cell = event.target.closest("td");
        highlightColumn(cell && grid.contains(cell) ? cell.cellIndex : -1);
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
            highlightColumn(-1);
        }
    });
})();
