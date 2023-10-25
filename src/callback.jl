"""
    compute_line_length(format_string)

Calculates the line length for the table format of a print callback.
"""
function compute_line_length(format_string)
    temp = strip(format_string, ['\n'])
    temp = replace(temp, "%" => "")
    temp = replace(temp, "e" => "")
    temp = replace(temp, "i" => "")
    temp = replace(temp, "s" => "")
    temp = split(temp, " ")
    len = sum(parse(Int, i) for i in temp)
    return len + 2 + length(temp) - 1
end

"""
    print_callback(state,storage)

Handles formating of the callback state into a table format with consistent length independent of state values.
"""
function print_callback(data, format_string; print_header=false, print_footer=false)
    print_formatted(fmt, args...) = @eval @printf($fmt, $(args...))

    if print_header || print_footer
        lenHeaderFooter = compute_line_length(format_string)
    end

    if print_footer
        line = "-"^lenHeaderFooter
        @printf("%s\e[s\n", line) # \e[s stores the cursor position at the end of the line
    elseif print_header
        line = "-"^lenHeaderFooter
        @printf("\n%s\n", line)
        s_format_string = replace(format_string, "e" => "s")
        s_format_string = replace(s_format_string, "i" => "s")
        print_formatted(s_format_string, data...)
        @printf("%s\e[s\n", line) # \e[s stores the cursor position at the end of the line    
    else
        print_formatted(format_string, data...)
    end
end

"""
    make_print_callback(callback, print_iter, headers, format_string, format_state)

Default verbose callback for fw_algorithms, that wraps around previous callback.
Prints the state to the console after print_iter iterations.
If the callback to be wrapped is of type nothing, always return true to enforce boolean output for non-nothing callbacks.
"""
function make_print_callback(callback, print_iter, headers, format_string, format_state)
    return function callback_with_prints(state, args...)
        if (state.tt == pp || state.tt == last)
            if state.t == 0 && state.tt == last
                print_callback(headers, format_string, print_header=true)
            end
            rep = format_state(state, args...)
            print_callback(rep, format_string)
            print_callback(nothing, format_string, print_footer=true)
            flush(stdout)
        elseif state.t == 1 ||
               mod(state.t, print_iter) == 0 ||
               state.tt == dualstep ||
               state.tt == last
            if state.t == 1
                state = @set state.tt = initial
                print_callback(headers, format_string, print_header=true)
            end
            rep = format_state(state, args...)
            extended_format_string = format_string[1:end-1] * "\e[s" * format_string[end] # Add escape code for storing cursor position before newline
            print_callback(rep, extended_format_string)
            flush(stdout)
        end

        if callback === nothing
            return true
        end
        return callback(state, args...)
    end
end

function make_print_callback_extension(callback, print_iter, headers, format_string, format_state)
    return function callback_with_prints(state, args...)
        if (state.tt == pp || state.tt == last)
            if state.t == 0 && state.tt == last
                print_callback(headers, format_string, print_header=true)
            end
            rep = format_state(state, args...)
            print("\e[u\e[1A\e[2D") # Move to end of "last"-row
            print_callback(rep, format_string)
            print("\e[u\e[2D") # Move to end of horizontal line
            print_callback(nothing, format_string, print_footer=true)
            flush(stdout)
        elseif state.t == 1 ||
               mod(state.t, print_iter) == 0 ||
               state.tt == dualstep ||
               state.tt == last
            if state.t == 1
                print("\e[u\e[3A") # Move to end of upper horizontal line
                line = "-"^compute_line_length(format_string)
                @printf("%s\n", line)
                print("\e[u\e[2A") # Move to end of the header row
                s_format_string = replace(format_string, "e" => "s")
                s_format_string = replace(s_format_string, "i" => "s")
                @eval @printf($s_format_string, $(headers...))
                print("\e[u\e[1A") # Move to end of lower horizontal line
                @printf("%s\n\n", line)
            end
            rep = format_state(state, args...)
            print("\e[u") # Move to end of table row
            print_callback(rep, format_string)
            flush(stdout)
        end

        if callback === nothing
            return true
        end
        return callback(state, args...)
    end
end


"""
    make_trajectory_callback(callback, traj_data)

Default trajectory logging callback for fw_algorithms, that wraps around previous callback.
Adds the state to the storage variable.
The state data is only the 5 first fields, usually:
`(t, primal, dual, dual_gap, time)`
If the callback to be wrapped is of type nothing, always return true to enforce boolean output for non-nothing callbacks.
"""
function make_trajectory_callback(callback, traj_data::Vector)
    return function callback_with_trajectory(state, args...)
        if state.tt !== last || state.tt !== pp
            push!(traj_data, callback_state(state))
        end
        if callback === nothing
            return true
        end
        return callback(state, args...)
    end
end
