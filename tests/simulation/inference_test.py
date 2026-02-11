import pytest
from simulation.inference import InferenceEngine

def build_action_1(**kwargs):
    kwargs['tracker'].append("action_1")

def build_action_2(**kwargs):
    kwargs['tracker'].append("action_2")

@pytest.fixture
def build_tracker():
    """Provides a list to track execution order of build actions."""
    return []

@pytest.fixture
def basic_engine():
    """Sets up a simple A -> B -> C dependency chain."""
    rules = [
        {"name": "root"},
        {
            "name": "mid", 
            "depends_on": ["root"], 
            "how_to_make": [build_action_1]
        },
        {
            "name": "target", 
            "depends_on": ["mid"], 
            "how_to_make": [build_action_2]
        }
    ]
    return InferenceEngine(rules)

def test_initialization(basic_engine):
    """Ensure the graph is built correctly with back-links."""
    assert "root" in basic_engine.entities
    assert "target" in basic_engine.entities
    # Check back-link: 'mid' should be a prerequisite for 'target'
    assert "target" in basic_engine.entities["mid"]._is_prerequisite

def test_make_execution_order(basic_engine, build_tracker):
    """Verify that 'make' executes steps in the correct dependency order."""
    # Mark the base as up-to-date
    basic_engine.entities["root"]._is_uptodate = True
    
    basic_engine.make("target", tracker=build_tracker)
    
    # Order should be depth-first: mid then target
    assert build_tracker == ["action_1", "action_2"]
    assert basic_engine.entities["target"]._is_uptodate is True

def test_idempotency(basic_engine, build_tracker):
    """Verify that calling make twice doesn't re-run actions if up-to-date."""
    basic_engine.entities["root"]._is_uptodate = True
    
    # First build
    basic_engine.make("target", tracker=build_tracker)
    assert len(build_tracker) == 2
    
    # Second build (should do nothing)
    basic_engine.make("target", tracker=build_tracker)
    assert len(build_tracker) == 2 

def test_invalidation(basic_engine, build_tracker):
    """Verify that invalidating a root triggers a rebuild of everything downstream."""
    basic_engine.entities["root"]._is_uptodate = True
    basic_engine.make("target", tracker=build_tracker)
    
    # Invalidate the very bottom
    basic_engine.invalidate("root")
    
    assert basic_engine.entities["mid"]._is_uptodate is False
    assert basic_engine.entities["target"]._is_uptodate is False
    
    # Clear tracker and rebuild
    build_tracker.clear()
    basic_engine.entities["root"]._is_uptodate = True
    basic_engine.make("target", tracker=build_tracker)
    assert len(build_tracker) == 2

def test_missing_dependency_error():
    """Verify that the engine raises KeyError for missing nodes."""
    rules = [{"name": "A", "depends_on": ["NON_EXISTENT"]}]
    with pytest.raises(KeyError, match="required by 'A' not found"):
        InferenceEngine(rules)

def test_invalid_target_error(basic_engine):
    """Verify error when trying to make a name not in the graph."""
    with pytest.raises(KeyError, match="unknown entity 'ghost'"):
        basic_engine.make("ghost")

def test_circular_dependency_error():
    """Verify that the engine detects loops like A -> B -> A."""
    rules = [
        {"name": "A", "depends_on": ["B"]},
        {"name": "B", "depends_on": ["A"]}
    ]
    # match ensures we are getting our specific cycle error
    with pytest.raises(ValueError, match="Circular dependency detected"):
        InferenceEngine(rules)